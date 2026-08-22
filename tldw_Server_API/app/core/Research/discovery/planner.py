"""Pure deterministic compiler for research discovery V2 plans."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date

from .contracts import (
    CREDENTIALED_ROUTE_SKIP_REASON,
    QUERY_MODE_NOT_SUPPORTED_SKIP_REASON,
    AccessRoute,
    BoundedDecimalQueryValuePolicy,
    BoundedTextQueryValuePolicy,
    BudgetCeilings,
    CredentialRequirement,
    DeferredNumericCSVQueryBinding,
    DiscoveryPlan,
    DispatchAllowance,
    DispatchIntent,
    ExactOrigin,
    ExactQueryValuePolicy,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
    OpaqueCursorQueryValuePolicy,
    OperationKind,
    PathSlot,
    PathSlotKind,
    PathTemplate,
    PlannedBudgetAllowance,
    PlannedDispatchGroup,
    PlannedLogicalAttempt,
    QueryMode,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RouteLimits,
    RoutePolicy,
    SkippedCode,
    SkippedStatus,
    SkippedTarget,
    SourceDefinition,
    SourcePredicate,
    budget_ceiling_violation,
    derive_plan_allowance,
)
from .registry import DiscoveryRegistry

PLANNER_VERSION = "research-discovery-planner-v2-foundation"
_DOI_REGISTRANT = re.compile(r"10\.[0-9]{4,9}\Z")
_CANONICAL_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_RFC3986_UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
_GENERAL_MAX_TERMS = 16
_GENERAL_MAX_TERM_CHARS = 64
_GENERAL_MAX_RAW_CHARS = _GENERAL_MAX_TERMS * _GENERAL_MAX_TERM_CHARS + _GENERAL_MAX_TERMS - 1
_GENERAL_MAX_RAW_UTF8_BYTES = _GENERAL_MAX_TERMS * _GENERAL_MAX_TERM_CHARS * 4 + _GENERAL_MAX_TERMS - 1
_PUBMED_ROUTE_ID = "pubmed_ncbi_eutils_pubmed_direct"
_PUBMED_BACKEND_ID = "ncbi_eutils_pubmed"
_PUBMED_ADAPTER_ID = "pubmed_v2"
_PUBMED_IDENTITY_POLICY_VERSION = "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21"
_PUBMED_IDENTITY_ADAPTER_VERSION = "pubmed-v2-ncbi-identity"
_PUBMED_IDENTITY_POLICY_DIGEST = "742b8aca76878ca06ab43ae17130627b5daaebea0a3c3ae25786521a9f159d22"
_PUBMED_IDENTITY_QUERY_KEYS = (
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
_NCBI_TOOL = "tldw_server"
_NCBI_EMAIL = "contact@tldwproject.com"
_PUBMED_CENTRAL_ROUTE_ID = "pubmed_central_esearch_summary_direct"
_PUBMED_CENTRAL_BACKEND_ID = "ncbi_eutils_pmc"
_PUBMED_CENTRAL_ADAPTER_ID = "pubmed_central_v2"
_PUBMED_CENTRAL_ADAPTER_VERSION = "pubmed-central-v2"
_PUBMED_CENTRAL_POLICY_VERSION = "research-discovery-route-policy-v2-clinicaltrials-pmc"
_PUBMED_CENTRAL_POLICY_DIGEST = "621115ce40342226999a120bfc3ab31fcac28a0e6eb2e37c39653bdd72791fc9"
_PUBMED_CENTRAL_BINDING_ID = "pmc_esearch_ids"
_CLINICALTRIALS_ROUTE_ID = "clinicaltrials_gov_studies_search_direct"
_CLINICALTRIALS_BACKEND_ID = "clinicaltrials_gov_api_v2"
_CLINICALTRIALS_ADAPTER_ID = "clinicaltrials_gov_v2"
_CLINICALTRIALS_ADAPTER_VERSION = "clinicaltrials-gov-v2"
_CLINICALTRIALS_POLICY_DIGEST = "80c6b86d91cb215477162138be4a7ea0a1935fbb40ae6c19e279106c258aab02"
_CLINICALTRIALS_FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,"
    "InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults"
)
_CLINICALTRIALS_QUERY_KEYS = (
    "query.term",
    "format",
    "markupFormat",
    "fields",
    "pageSize",
    "countTotal",
    "pageToken",
)


def _is_identity_pubmed_route(route: AccessRoute) -> bool:
    """Return whether one route is the complete sealed NCBI identity overlay."""
    if type(route) is not AccessRoute or type(route.policy) is not RoutePolicy:
        return False
    return (
        route.route_id == _PUBMED_ROUTE_ID
        and route.backend_id == _PUBMED_BACKEND_ID
        and route.adapter_id == _PUBMED_ADAPTER_ID
        and route.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION
        and route.policy.policy_version == _PUBMED_IDENTITY_POLICY_VERSION
        and _has_exact_pubmed_identity_policy(route)
    )


def _is_foundation_pubmed_policy_owner(route: AccessRoute) -> bool:
    """Exclude the exact foundation identity from overlay-specific sealing."""
    if type(route) is not AccessRoute or type(route.policy) is not RoutePolicy:
        return False
    return (
        route.route_id == _PUBMED_ROUTE_ID
        and route.backend_id == _PUBMED_BACKEND_ID
        and route.adapter_id == _PUBMED_ADAPTER_ID
        and route.adapter_version == "foundation-v2"
        and route.policy.policy_version == "research-discovery-route-policy-v2-foundation"
    )


def _has_pubmed_identity_component(route: AccessRoute) -> bool:
    """Return whether a non-foundation route claims any PubMed overlay marker."""
    if type(route) is not AccessRoute:
        return False
    if type(route.policy) is not RoutePolicy:
        return route.route_id == _PUBMED_ROUTE_ID
    if _is_foundation_pubmed_policy_owner(route):
        return False
    return any(
        (
            route.route_id == _PUBMED_ROUTE_ID,
            route.backend_id == _PUBMED_BACKEND_ID,
            route.adapter_id == _PUBMED_ADAPTER_ID,
            route.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION,
            route.policy.policy_version == _PUBMED_IDENTITY_POLICY_VERSION,
        )
    )


def _is_pubmed_central_route(route: AccessRoute) -> bool:
    """Return whether one route has the complete sealed PMC identity tuple."""
    if type(route) is not AccessRoute or type(route.policy) is not RoutePolicy:
        return False
    return (
        type(route.route_id) is str
        and route.route_id == _PUBMED_CENTRAL_ROUTE_ID
        and type(route.backend_id) is str
        and route.backend_id == _PUBMED_CENTRAL_BACKEND_ID
        and type(route.adapter_id) is str
        and route.adapter_id == _PUBMED_CENTRAL_ADAPTER_ID
        and type(route.adapter_version) is str
        and route.adapter_version == _PUBMED_CENTRAL_ADAPTER_VERSION
        and type(route.policy.policy_version) is str
        and route.policy.policy_version == _PUBMED_CENTRAL_POLICY_VERSION
    )


def _has_pubmed_central_identity_component(route: AccessRoute) -> bool:
    if _has_exact_clinicaltrials_policy(route):
        return False
    try:
        policy_version = route.policy.policy_version
    except Exception:  # noqa: BLE001 - malformed registry policies fail closed as identity drift.
        policy_version = None
    return any(
        (
            route.route_id == _PUBMED_CENTRAL_ROUTE_ID,
            route.backend_id == _PUBMED_CENTRAL_BACKEND_ID,
            route.adapter_id == _PUBMED_CENTRAL_ADAPTER_ID,
            route.adapter_version == _PUBMED_CENTRAL_ADAPTER_VERSION,
            route.route_id == _CLINICALTRIALS_ROUTE_ID,
            route.backend_id == _CLINICALTRIALS_BACKEND_ID,
            route.adapter_id == _CLINICALTRIALS_ADAPTER_ID,
            route.adapter_version == _CLINICALTRIALS_ADAPTER_VERSION,
            policy_version == _PUBMED_CENTRAL_POLICY_VERSION,
        )
    )


def _is_clinicaltrials_shared_policy_owner(route: AccessRoute) -> bool:
    """Exclude only the complete exact sibling route that owns the shared marker."""
    return _has_exact_clinicaltrials_policy(route)


def _is_exact_contract_enum_member(
    value: object,
    *,
    enum_name: str,
    member_name: str,
    member_value: str,
) -> bool:
    """Match one sealed contracts enum member without importing family-owned code."""
    value_type = type(value)
    try:
        return (
            value_type.__module__ == "tldw_Server_API.app.core.Research.discovery.contracts"
            and value_type.__qualname__ == enum_name
            and value_type.__members__[member_name] is value
            and type(value.value) is str
            and value.value == member_value
        )
    except (AttributeError, KeyError, TypeError):
        return False


def _is_exact_string_tuple(value: object, expected: tuple[str, ...]) -> bool:
    """Match tuple shape, values, and exact scalar types."""
    return (
        type(value) is tuple
        and len(value) == len(expected)
        and all(type(actual) is str and actual == approved for actual, approved in zip(value, expected))
    )


def _has_exact_exact_query_value_policy(value: object, name: str, exact_value: str) -> bool:
    return (
        type(value) is ExactQueryValuePolicy
        and type(value.name) is str
        and value.name == name
        and type(value.value) is str
        and value.value == exact_value
        and type(value.required) is bool
        and value.required is True
    )


def _has_exact_clinicaltrials_query_value_policies(value: object) -> bool:
    if type(value) is not tuple or len(value) != 7:
        return False
    literal_terms, format_rule, markup_rule, fields_rule, page_size, count_total, page_token = value
    return (
        type(literal_terms) is LiteralTermsQueryValuePolicy
        and type(literal_terms.name) is str
        and literal_terms.name == "query.term"
        and type(literal_terms.fixed_suffix) is str
        and literal_terms.fixed_suffix == ""
        and type(literal_terms.max_terms) is int
        and literal_terms.max_terms == 8
        and type(literal_terms.max_term_chars) is int
        and literal_terms.max_term_chars == 32
        and type(literal_terms.required) is bool
        and literal_terms.required is True
        and _has_exact_exact_query_value_policy(format_rule, "format", "json")
        and _has_exact_exact_query_value_policy(markup_rule, "markupFormat", "legacy")
        and _has_exact_exact_query_value_policy(fields_rule, "fields", _CLINICALTRIALS_FIELDS)
        and type(page_size) is BoundedDecimalQueryValuePolicy
        and type(page_size.name) is str
        and page_size.name == "pageSize"
        and type(page_size.maximum) is int
        and page_size.maximum == 50
        and type(page_size.required) is bool
        and page_size.required is True
        and _has_exact_exact_query_value_policy(count_total, "countTotal", "true")
        and type(page_token) is OpaqueCursorQueryValuePolicy
        and type(page_token.name) is str
        and page_token.name == "pageToken"
        and type(page_token.max_chars) is int
        and page_token.max_chars == 1_024
        and type(page_token.required) is bool
        and page_token.required is False
    )


def _has_exact_clinicaltrials_limits(limits: object) -> bool:
    if type(limits) is not RouteLimits:
        return False
    values = (
        limits.max_pages,
        limits.max_redirects,
        limits.max_retries,
        limits.timeout_ms,
        limits.max_response_bytes,
        limits.max_results,
        limits.max_request_body_bytes,
    )
    return all(type(value) is int for value in values) and values == (
        2,
        0,
        0,
        20_000,
        2_097_152,
        100,
        16_384,
    )


def _has_exact_clinicaltrials_policy(route: AccessRoute) -> bool:
    """Anchor the shared ClinicalTrials marker to one complete approved route."""
    if type(route) is not AccessRoute or type(route.policy) is not RoutePolicy:
        return False
    policy = route.policy
    origin = policy.origin
    return (
        type(route.route_id) is str
        and route.route_id == _CLINICALTRIALS_ROUTE_ID
        and type(route.backend_id) is str
        and route.backend_id == _CLINICALTRIALS_BACKEND_ID
        and type(route.adapter_id) is str
        and route.adapter_id == _CLINICALTRIALS_ADAPTER_ID
        and type(route.adapter_version) is str
        and route.adapter_version == _CLINICALTRIALS_ADAPTER_VERSION
        and _is_exact_contract_enum_member(
            route.route_kind,
            enum_name="RouteKind",
            member_name="DIRECT",
            member_value="direct",
        )
        and type(route.query_modes) is tuple
        and len(route.query_modes) == 1
        and route.query_modes[0] is QueryMode.GENERAL_FREE_TEXT
        and _is_exact_contract_enum_member(
            route.source_constraint,
            enum_name="SourceConstraint",
            member_name="NATIVE_CORPUS",
            member_value="native_corpus",
        )
        and type(route.attribution_basis) is str
        and route.attribution_basis == "native_nct_record"
        and route.credential_requirement is CredentialRequirement.NONE
        and type(route.fallback_order) is int
        and route.fallback_order == 0
        and type(route.max_physical_dispatches) is int
        and route.max_physical_dispatches == 2
        and type(policy.policy_version) is str
        and policy.policy_version == _PUBMED_CENTRAL_POLICY_VERSION
        and type(origin) is ExactOrigin
        and type(origin.scheme) is str
        and origin.scheme == "https"
        and type(origin.host) is str
        and origin.host == "clinicaltrials.gov"
        and type(origin.port) is int
        and origin.port == 443
        and type(policy.policy_digest) is str
        and policy.policy_digest == _CLINICALTRIALS_POLICY_DIGEST
        and _is_exact_string_tuple(policy.methods, ("GET",))
        and _is_exact_string_tuple(policy.paths, ("/api/v2/studies",))
        and policy.path_template is None
        and _is_exact_string_tuple(policy.allowed_query_keys, _CLINICALTRIALS_QUERY_KEYS)
        and type(policy.pagination_query_key) is str
        and policy.pagination_query_key == "pageToken"
        and policy.pagination_json_body_key is None
        and type(policy.allowed_json_body_keys) is tuple
        and policy.allowed_json_body_keys == ()
        and type(policy.integer_json_body_keys) is tuple
        and policy.integer_json_body_keys == ()
        and _has_exact_clinicaltrials_query_value_policies(policy.query_value_policies)
        and _has_exact_clinicaltrials_limits(policy.limits)
    )


def _has_exact_pubmed_identity_policy(route: AccessRoute) -> bool:
    """Anchor the PubMed identity overlay to one approved route and policy."""
    policy = route.policy
    origin = policy.origin
    return (
        type(route) is AccessRoute
        and type(route.route_id) is str
        and route.route_id == _PUBMED_ROUTE_ID
        and type(route.backend_id) is str
        and route.backend_id == _PUBMED_BACKEND_ID
        and type(route.adapter_id) is str
        and route.adapter_id == _PUBMED_ADAPTER_ID
        and type(route.adapter_version) is str
        and route.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION
        and _is_exact_contract_enum_member(
            route.route_kind,
            enum_name="RouteKind",
            member_name="DIRECT",
            member_value="direct",
        )
        and type(route.query_modes) is tuple
        and len(route.query_modes) == 1
        and route.query_modes[0] is QueryMode.STRUCTURED_QUERY
        and _is_exact_contract_enum_member(
            route.source_constraint,
            enum_name="SourceConstraint",
            member_name="NATIVE_CORPUS",
            member_value="native_corpus",
        )
        and type(route.attribution_basis) is str
        and route.attribution_basis == "native_response"
        and route.credential_requirement is CredentialRequirement.NONE
        and type(route.fallback_order) is int
        and route.fallback_order == 0
        and type(route.max_physical_dispatches) is int
        and route.max_physical_dispatches == 2
        and type(policy.policy_version) is str
        and policy.policy_version == _PUBMED_IDENTITY_POLICY_VERSION
        and type(origin) is ExactOrigin
        and type(origin.scheme) is str
        and origin.scheme == "https"
        and type(origin.host) is str
        and origin.host == "eutils.ncbi.nlm.nih.gov"
        and type(origin.port) is int
        and origin.port == 443
        and type(policy.policy_digest) is str
        and policy.policy_digest == _PUBMED_IDENTITY_POLICY_DIGEST
        and _is_exact_string_tuple(policy.methods, ("GET",))
        and _is_exact_string_tuple(
            policy.paths,
            ("/entrez/eutils/esearch.fcgi", "/entrez/eutils/esummary.fcgi"),
        )
        and policy.path_template is None
        and _is_exact_string_tuple(policy.allowed_query_keys, _PUBMED_IDENTITY_QUERY_KEYS)
        and type(policy.pagination_query_key) is str
        and policy.pagination_query_key == "retstart"
        and policy.pagination_json_body_key is None
        and type(policy.allowed_json_body_keys) is tuple
        and policy.allowed_json_body_keys == ()
        and type(policy.integer_json_body_keys) is tuple
        and policy.integer_json_body_keys == ()
        and type(policy.query_value_policies) is tuple
        and policy.query_value_policies == ()
        and _has_exact_pubmed_identity_limits(policy.limits)
    )


def _has_exact_pubmed_identity_limits(limits: object) -> bool:
    if type(limits) is not RouteLimits:
        return False
    values = (
        limits.max_pages,
        limits.max_redirects,
        limits.max_retries,
        limits.timeout_ms,
        limits.max_response_bytes,
        limits.max_results,
        limits.max_request_body_bytes,
    )
    return all(type(value) is int for value in values) and values == (
        1,
        0,
        0,
        20_000,
        2_097_152,
        100,
        16_384,
    )


def _has_exact_pubmed_central_policy(route: AccessRoute) -> bool:
    if type(route) is not AccessRoute or type(route.policy) is not RoutePolicy:
        return False
    policy = route.policy
    origin = policy.origin
    return (
        _is_pubmed_central_route(route)
        and _is_exact_contract_enum_member(
            route.route_kind,
            enum_name="RouteKind",
            member_name="DIRECT",
            member_value="direct",
        )
        and type(route.query_modes) is tuple
        and len(route.query_modes) == 1
        and route.query_modes[0] is QueryMode.GENERAL_FREE_TEXT
        and _is_exact_contract_enum_member(
            route.source_constraint,
            enum_name="SourceConstraint",
            member_name="NATIVE_CORPUS",
            member_value="native_corpus",
        )
        and type(route.attribution_basis) is str
        and route.attribution_basis == "ncbi_pmc_database"
        and route.credential_requirement is CredentialRequirement.NONE
        and type(route.max_physical_dispatches) is int
        and route.max_physical_dispatches == 2
        and type(route.fallback_order) is int
        and route.fallback_order == 0
        and type(policy.policy_version) is str
        and policy.policy_version == _PUBMED_CENTRAL_POLICY_VERSION
        and type(origin) is ExactOrigin
        and type(origin.scheme) is str
        and origin.scheme == "https"
        and type(origin.host) is str
        and origin.host == "eutils.ncbi.nlm.nih.gov"
        and type(origin.port) is int
        and origin.port == 443
        and type(policy.policy_digest) is str
        and policy.policy_digest == _PUBMED_CENTRAL_POLICY_DIGEST
        and _is_exact_string_tuple(policy.methods, ("GET",))
        and _is_exact_string_tuple(
            policy.paths,
            ("/entrez/eutils/esearch.fcgi", "/entrez/eutils/esummary.fcgi"),
        )
        and policy.path_template is None
        and _is_exact_string_tuple(
            policy.allowed_query_keys,
            ("db", "term", "retstart", "retmax", "retmode", "tool", "email", "id"),
        )
        and type(policy.pagination_query_key) is str
        and policy.pagination_query_key == "retstart"
        and policy.pagination_json_body_key is None
        and type(policy.allowed_json_body_keys) is tuple
        and policy.allowed_json_body_keys == ()
        and type(policy.integer_json_body_keys) is tuple
        and policy.integer_json_body_keys == ()
        and type(policy.query_value_policies) is tuple
        and policy.query_value_policies == ()
        and _has_exact_pubmed_central_limits(policy.limits)
    )


def _has_exact_pubmed_central_limits(limits: object) -> bool:
    if type(limits) is not RouteLimits:
        return False
    values = (
        limits.max_pages,
        limits.max_redirects,
        limits.max_retries,
        limits.timeout_ms,
        limits.max_response_bytes,
        limits.max_results,
        limits.max_request_body_bytes,
    )
    return all(type(value) is int for value in values) and values == (
        1,
        0,
        0,
        20_000,
        2_097_152,
        100,
        16_384,
    )


def _planning_error(code: str) -> ValueError:
    """Build a centralized planning error without loading application services on import."""
    from tldw_Server_API.app.core.exceptions import PlanningError

    return PlanningError(code)


def __getattr__(name: str) -> object:
    """Resolve the legacy planning-error export without breaking planner purity."""
    if name != "PlanningError":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tldw_Server_API.app.core.exceptions import PlanningError

    return PlanningError


@dataclass(frozen=True, slots=True)
class GeneralFreeTextQuery:
    """Literal Unicode terms for a route-owned free-text expression."""

    text: str


@dataclass(frozen=True, slots=True)
class IdentifierLookupQuery:
    """One canonical ASCII DOI to resolve through an identifier route."""

    doi: str


@dataclass(frozen=True, slots=True)
class DateIntervalQuery:
    """One bounded inclusive date interval with an optional category."""

    start_date: str
    end_date: str
    category: str | None = None


PlanningQuery = str | GeneralFreeTextQuery | IdentifierLookupQuery | DateIntervalQuery
_PLANNING_QUERY_TYPES = (str, GeneralFreeTextQuery, IdentifierLookupQuery, DateIntervalQuery)


@dataclass(frozen=True, slots=True)
class _NormalizedPlanningQuery:
    """Planner-owned normalized values for one exact public query type."""

    mode: QueryMode
    normalized_query: str
    terms: tuple[str, ...] = ()
    doi_registrant: str | None = None
    doi_suffix: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Explicit V2 selection and normalized planning inputs."""

    source_ids: tuple[str, ...]
    query: PlanningQuery
    filters: tuple[QueryPair, ...]
    result_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_ids, tuple):
            raise TypeError("source_ids_must_be_tuple")
        if not self.source_ids or any(
            not isinstance(source_id, str) or not source_id.strip() for source_id in self.source_ids
        ):
            raise ValueError("explicit_selection_requires_source_ids")
        if type(self.query) not in _PLANNING_QUERY_TYPES:
            raise TypeError("query_must_be_exact_planning_query")
        _normalize_planning_query(self.query)
        if not isinstance(self.filters, tuple) or any(not isinstance(item, QueryPair) for item in self.filters):
            raise TypeError("filters_must_be_query_pair_tuple")
        if type(self.query) is not str and self.filters:
            raise ValueError("typed_query_filters_not_supported")
        if type(self.result_limit) is not int or self.result_limit <= 0:
            raise ValueError("result_limit_must_be_positive")


@dataclass(frozen=True, slots=True)
class _LogicalAttempt:
    source_priority: int
    route: AccessRoute
    catalog_source_id: str
    selection_reason: str
    source_predicate: SourcePredicate | None
    normalized_query: str
    filters: tuple[QueryPair, ...]
    intents: tuple[DispatchIntent, ...]
    allowance: DispatchAllowance


def _copy_source_predicate(predicate: SourcePredicate | None) -> SourcePredicate | None:
    """Reconstruct one registry predicate for an independent logical attempt."""
    if predicate is None:
        return None
    return SourcePredicate(
        field_path=predicate.field_path,
        operator=predicate.operator,
        values=predicate.values,
        case_sensitive=predicate.case_sensitive,
    )


def compile_discovery_plan(
    request: PlanningRequest,
    *,
    registry: DiscoveryRegistry,
    readiness: ReadinessOverlay,
    budget: BudgetCeilings,
) -> DiscoveryPlan:
    """Compile explicit source intent into a stable bounded plan."""
    if not isinstance(request, PlanningRequest):
        raise TypeError("request_must_be_planning_request")
    if not isinstance(registry, DiscoveryRegistry):
        raise TypeError("registry_must_be_discovery_registry")
    if not isinstance(readiness, ReadinessOverlay):
        raise TypeError("readiness_must_be_readiness_overlay")
    if not isinstance(budget, BudgetCeilings):
        raise TypeError("budget_must_be_budget_ceilings")
    _validate_readiness_references(registry, readiness)

    query_context = _normalize_planning_query(request.query)
    normalized_query = query_context.normalized_query
    filters = tuple(sorted(request.filters, key=lambda item: (item.name, item.value)))
    sources = _resolve_sources(request.source_ids, registry)
    logical: list[_LogicalAttempt] = []
    skipped: list[SkippedTarget] = []

    for source in sources:
        references = {reference.route_id: reference for reference in source.route_references}
        for route in registry.routes_for_source(source.catalog_source_id):
            if _has_pubmed_central_identity_component(route) and (
                not _is_pubmed_central_route(route) or not _has_exact_pubmed_central_policy(route)
            ):
                raise _planning_error(f"invalid_pubmed_central_route_identity:{route.route_id}")
            if _has_pubmed_identity_component(route) and not _is_identity_pubmed_route(route):
                raise _planning_error(f"invalid_pubmed_route_identity:{route.route_id}")
            if query_context.mode not in route.query_modes:
                skipped.append(
                    SkippedTarget(
                        requested_source_id=source.catalog_source_id,
                        route_id=route.route_id,
                        status=SkippedStatus.SKIPPED,
                        code=SkippedCode.QUERY_MODE_NOT_SUPPORTED,
                        reason=QUERY_MODE_NOT_SUPPORTED_SKIP_REASON,
                    )
                )
                continue
            route_readiness = readiness.get(route.route_id)
            if route_readiness is None:
                raise _planning_error(f"missing_readiness:{route.route_id}")
            if route.credential_requirement is not CredentialRequirement.NONE:
                skipped.append(
                    SkippedTarget(
                        requested_source_id=source.catalog_source_id,
                        route_id=route.route_id,
                        status=SkippedStatus.UNAVAILABLE,
                        code=SkippedCode.CREDENTIALED_OUT_OF_SCOPE,
                        reason=CREDENTIALED_ROUTE_SKIP_REASON,
                    )
                )
                continue
            if route_readiness.state is not ReadinessState.READY:
                skipped.append(
                    SkippedTarget(
                        requested_source_id=source.catalog_source_id,
                        route_id=route.route_id,
                        status=SkippedStatus.SKIPPED,
                        code=SkippedCode.ROUTE_NOT_READY,
                        reason=route_readiness.reason,
                    )
                )
                continue
            if _is_identity_pubmed_route(route) and any(filter_.name in {"tool", "email"} for filter_ in filters):
                raise _planning_error(f"identity_query_filter_not_allowed:{route.route_id}")
            if _is_pubmed_central_route(route) and any(
                filter_.name in {"tool", "email", "sort"} for filter_ in filters
            ):
                raise _planning_error(f"pmc_query_filter_not_allowed:{route.route_id}")

            intents = (
                _build_intents(route, normalized_query, request.result_limit)
                if type(request.query) is str
                else _build_typed_intents(route, query_context, request.result_limit)
            )
            logical.append(
                _LogicalAttempt(
                    source_priority=source.priority,
                    route=route,
                    catalog_source_id=source.catalog_source_id,
                    selection_reason="explicit",
                    source_predicate=_copy_source_predicate(references[route.route_id].source_predicate),
                    normalized_query=normalized_query,
                    filters=filters,
                    intents=intents,
                    allowance=DispatchAllowance(
                        physical_dispatches=route.max_physical_dispatches,
                        pages=route.policy.limits.max_pages,
                        redirects=route.policy.limits.max_redirects,
                        retries=route.policy.limits.max_retries,
                    ),
                )
            )
    dispatch_groups = _coalesce(logical)
    allowance = derive_plan_allowance(dispatch_groups, request.result_limit)
    _enforce_budget(allowance, budget)
    return DiscoveryPlan(
        planner_version=PLANNER_VERSION,
        catalog_version=registry.catalog_version,
        registry_version=registry.registry_version,
        readiness_version=readiness.overlay_version,
        execution_mode=readiness.execution_mode,
        normalized_query=normalized_query,
        filters=filters,
        result_limit=request.result_limit,
        dispatch_groups=dispatch_groups,
        skipped=tuple(skipped),
        ceilings=budget,
    )


def canonical_plan_bytes(plan: DiscoveryPlan) -> bytes:
    """Serialize a plan deterministically for persistence and comparison."""
    if not isinstance(plan, DiscoveryPlan):
        raise TypeError("plan_must_be_discovery_plan")
    return _canonical_json(asdict(plan))


def _resolve_sources(
    source_ids: tuple[str, ...],
    registry: DiscoveryRegistry,
) -> tuple[SourceDefinition, ...]:
    resolved: dict[str, SourceDefinition] = {}
    for source_id in source_ids:
        try:
            source = registry.get_source(source_id)
        except KeyError:
            raise _planning_error(f"unknown_source:{source_id}") from None
        resolved[source.catalog_source_id] = source
    return tuple(sorted(resolved.values(), key=lambda source: (source.priority, source.catalog_source_id)))


def _normalize_planning_query(query: PlanningQuery) -> _NormalizedPlanningQuery:
    """Validate one exact query type and return its route-selection values."""
    if type(query) is str:
        if not query.strip():
            raise _planning_error("query_must_be_nonempty")
        return _NormalizedPlanningQuery(QueryMode.STRUCTURED_QUERY, _normalize_query(query))
    if type(query) is GeneralFreeTextQuery:
        if type(query.text) is not str:
            raise _planning_error("general_query_text_must_be_exact_string")
        if len(query.text) > _GENERAL_MAX_RAW_CHARS:
            raise _planning_error("general_query_input_limit_exceeded")
        try:
            raw_utf8_bytes = len(query.text.encode("utf-8"))
        except UnicodeEncodeError:
            raise _planning_error("general_query_contains_invalid_unicode") from None
        if raw_utf8_bytes > _GENERAL_MAX_RAW_UTF8_BYTES:
            raise _planning_error("general_query_input_limit_exceeded")
        if any(unicodedata.category(character).startswith("C") for character in query.text):
            raise _planning_error("general_query_contains_invalid_unicode")
        canonical = unicodedata.normalize("NFKC", query.text)
        terms = _unicode_alphanumeric_terms(canonical)
        if not terms:
            raise _planning_error("general_query_requires_term")
        return _NormalizedPlanningQuery(
            QueryMode.GENERAL_FREE_TEXT,
            " ".join(terms),
            terms=terms,
        )
    if type(query) is IdentifierLookupQuery:
        if type(query.doi) is not str:
            raise _planning_error("doi_must_be_exact_string")
        if not query.doi.isascii() or query.doi.count("/") != 1:
            raise _planning_error("invalid_doi")
        registrant, suffix = query.doi.split("/", 1)
        if (
            _DOI_REGISTRANT.fullmatch(registrant) is None
            or not suffix
            or len(suffix) > 128
            or not suffix[0].isalnum()
            or any(character in " /\\%?#" or not "!" <= character <= "~" for character in suffix)
        ):
            raise _planning_error("invalid_doi")
        canonical_doi = query.doi.lower()
        registrant, suffix = canonical_doi.split("/", 1)
        return _NormalizedPlanningQuery(
            QueryMode.IDENTIFIER_LOOKUP,
            canonical_doi,
            doi_registrant=registrant,
            doi_suffix=suffix,
        )
    if type(query) is DateIntervalQuery:
        start = _validated_date(query.start_date)
        end = _validated_date(query.end_date)
        if start > end or (end - start).days + 1 > 366:
            raise _planning_error("invalid_date_interval")
        category = query.category
        if category is not None:
            if (
                type(category) is not str
                or not category
                or len(category) > 128
                or unicodedata.normalize("NFKC", category) != category
                or category != category.strip()
                or "  " in category
                or not any(character.isalnum() for character in category)
                or any(not character.isalnum() and character not in " -&/" for character in category)
            ):
                raise _planning_error("invalid_category")
        return _NormalizedPlanningQuery(
            QueryMode.CATEGORY_BROWSE if category is not None else QueryMode.DATE_INTERVAL,
            f"{query.start_date}/{query.end_date}" + (f"/{category}" if category is not None else ""),
            start_date=query.start_date,
            end_date=query.end_date,
            category=category,
        )
    raise TypeError("query_must_be_exact_planning_query")


def _unicode_alphanumeric_terms(value: str) -> tuple[str, ...]:
    """Split canonical Unicode text into contiguous literal alphanumeric terms."""
    terms: list[str] = []
    current: list[str] = []
    for character in value:
        if character.isalnum():
            if not current and len(terms) == _GENERAL_MAX_TERMS:
                raise _planning_error("general_query_term_limit_exceeded")
            if len(current) == _GENERAL_MAX_TERM_CHARS:
                raise _planning_error("general_query_term_limit_exceeded")
            current.append(character)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return tuple(terms)


def _validated_date(value: object) -> date:
    """Return one exact canonical ISO calendar date or fail planning."""
    if type(value) is not str or _CANONICAL_DATE.fullmatch(value) is None:
        raise _planning_error("invalid_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise _planning_error("invalid_date") from None
    if parsed.isoformat() != value:
        raise _planning_error("invalid_date")
    return parsed


def _normalize_query(query: str) -> str:
    try:
        query.encode("utf-8")
    except UnicodeEncodeError:
        raise _planning_error("query_contains_invalid_unicode") from None
    return " ".join(unicodedata.normalize("NFKC", query).split()).casefold()


def _build_typed_intents(
    route: AccessRoute,
    query: _NormalizedPlanningQuery,
    result_limit: int,
) -> tuple[DispatchIntent, ...]:
    """Render typed query values only through closed route policy shapes."""
    if route.policy.allowed_json_body_keys:
        raise _planning_error(f"typed_intent_json_body_not_supported:{route.route_id}")
    if query.mode is QueryMode.GENERAL_FREE_TEXT:
        if _has_pubmed_central_identity_component(route):
            if not _is_pubmed_central_route(route) or not _has_exact_pubmed_central_policy(route):
                raise _planning_error(f"invalid_pubmed_central_route_identity:{route.route_id}")
            limit = min(result_limit, route.policy.limits.max_results)
            expression = " AND ".join(f'"{term}"' for term in query.terms)
            return (
                _intent(
                    route,
                    OperationKind.SEARCH,
                    route.policy.paths[0],
                    (
                        QueryPair("db", "pmc"),
                        QueryPair("term", expression),
                        QueryPair("retstart", "0"),
                        QueryPair("retmax", str(limit)),
                        QueryPair("retmode", "json"),
                        QueryPair("tool", _NCBI_TOOL),
                        QueryPair("email", _NCBI_EMAIL),
                    ),
                ),
                _intent(
                    route,
                    OperationKind.CONDITIONAL_SUMMARY,
                    route.policy.paths[1],
                    (
                        QueryPair("db", "pmc"),
                        QueryPair("retmode", "json"),
                        QueryPair("tool", _NCBI_TOOL),
                        QueryPair("email", _NCBI_EMAIL),
                    ),
                    query_bindings=(
                        DeferredNumericCSVQueryBinding(
                            binding_id=_PUBMED_CENTRAL_BINDING_ID,
                            query_name="id",
                            max_items=limit,
                            max_item_chars=16,
                        ),
                    ),
                ),
            )
        if route.policy.path_template is not None or len(route.policy.paths) != 1:
            raise _planning_error(f"invalid_general_query_path_policy:{route.route_id}")
        policies = {policy.name: policy for policy in route.policy.query_value_policies}
        pairs: list[QueryPair] = []
        literal_terms_seen = False
        omitted_opaque = 0
        for name in route.policy.allowed_query_keys:
            policy = policies.get(name)
            if type(policy) is OpaqueCursorQueryValuePolicy:
                if policy.required or name != route.policy.pagination_query_key or omitted_opaque:
                    raise _planning_error(f"invalid_optional_opaque_cursor_policy:{route.route_id}")
                omitted_opaque += 1
                continue
            if type(policy) is LiteralTermsQueryValuePolicy:
                if (
                    literal_terms_seen
                    or not 1 <= len(query.terms) <= policy.max_terms
                    or any(len(term) > policy.max_term_chars for term in query.terms)
                ):
                    raise _planning_error(f"invalid_literal_terms_policy:{route.route_id}")
                literal_terms_seen = True
                pairs.append(
                    QueryPair(
                        name,
                        " AND ".join(f'"{term}"' for term in query.terms) + policy.fixed_suffix,
                    )
                )
            elif type(policy) is ExactQueryValuePolicy:
                pairs.append(QueryPair(name, policy.value))
            elif type(policy) is BoundedDecimalQueryValuePolicy:
                pairs.append(QueryPair(name, str(min(result_limit, route.policy.limits.max_results, policy.maximum))))
            else:
                raise _planning_error(f"invalid_general_query_value_policy:{route.route_id}")
        if not literal_terms_seen or len(pairs) + omitted_opaque != len(route.policy.allowed_query_keys):
            raise _planning_error(f"incomplete_general_query_policy:{route.route_id}")
        return (
            _intent(
                route,
                OperationKind.SEARCH,
                route.policy.paths[0],
                tuple(pairs),
            ),
        )

    template = route.policy.path_template
    if type(template) is not PathTemplate:
        raise _planning_error(f"typed_query_requires_path_template:{route.route_id}")
    if query.mode is QueryMode.IDENTIFIER_LOOKUP:
        if (
            query.doi_registrant is None
            or query.doi_suffix is None
            or template.pagination_segment_index is not None
            or route.policy.allowed_query_keys
            or route.policy.query_value_policies
        ):
            raise _planning_error(f"invalid_identifier_route_policy:{route.route_id}")
        path = _render_path_template(
            template,
            {
                PathSlotKind.DOI_REGISTRANT: (query.doi_registrant,),
                PathSlotKind.DOI_SUFFIX: (query.doi_suffix,),
            },
        )
        return (_intent(route, OperationKind.SEARCH, path, ()),)

    if query.mode not in {QueryMode.DATE_INTERVAL, QueryMode.CATEGORY_BROWSE}:
        raise _planning_error(f"unsupported_typed_query_mode:{route.route_id}")
    if query.start_date is None or query.end_date is None:
        raise _planning_error(f"invalid_interval_query:{route.route_id}")
    path = _render_path_template(
        template,
        {
            PathSlotKind.DATE: (query.start_date, query.end_date),
            PathSlotKind.UINT: ("0",),
        },
    )
    policies = route.policy.query_value_policies
    if (
        len(policies) != 1
        or type(policies[0]) is not BoundedTextQueryValuePolicy
        or policies[0].name != "category"
        or route.policy.allowed_query_keys != ("category",)
    ):
        raise _planning_error(f"invalid_interval_query_policy:{route.route_id}")
    category_policy = policies[0]
    if query.category is None:
        if category_policy.required:
            raise _planning_error(f"interval_category_required:{route.route_id}")
        pairs = ()
    else:
        if len(query.category) > category_policy.max_chars:
            raise _planning_error(f"interval_category_exceeds_policy:{route.route_id}")
        pairs = (QueryPair("category", query.category),)
    return (_intent(route, OperationKind.SEARCH, path, pairs),)


def _render_path_template(
    template: PathTemplate,
    values: dict[PathSlotKind, tuple[str, ...]],
) -> str:
    """Render exact typed slot values into one canonical encoded path."""
    remaining = {kind: list(items) for kind, items in values.items()}
    rendered: list[str] = []
    for segment in template.segments:
        if type(segment) is str:
            rendered.append(segment)
            continue
        if type(segment) is not PathSlot or not remaining.get(segment.kind):
            raise _planning_error("path_template_shape_mismatch")
        value = remaining[segment.kind].pop(0)
        if not value.isascii() or len(value) > segment.max_chars:
            raise _planning_error("path_template_value_invalid")
        rendered.append(_encode_ascii_path_segment(value))
    if any(items for items in remaining.values()):
        raise _planning_error("path_template_shape_mismatch")
    return f"/{'/'.join(rendered)}"


def _encode_ascii_path_segment(value: str) -> str:
    """Percent-encode one validated ASCII segment without transport imports."""
    return "".join(character if character in _RFC3986_UNRESERVED else f"%{ord(character):02X}" for character in value)


def _build_intents(
    route: AccessRoute,
    normalized_query: str,
    result_limit: int,
) -> tuple[DispatchIntent, ...]:
    if _has_pubmed_central_identity_component(route):
        raise _planning_error(f"invalid_pubmed_central_route_identity:{route.route_id}")
    limit = min(result_limit, route.policy.limits.max_results)
    foundation_pubmed = (
        route.route_id == "pubmed_ncbi_eutils_pubmed_direct"
        and route.backend_id == "ncbi_eutils_pubmed"
        and route.adapter_id == "pubmed_v2"
        and route.adapter_version == "foundation-v2"
        and route.policy.policy_version == "research-discovery-route-policy-v2-foundation"
    )
    identity_pubmed = _is_identity_pubmed_route(route)
    if foundation_pubmed or identity_pubmed:
        pairs = (
            QueryPair("db", "pubmed"),
            QueryPair("term", normalized_query),
            QueryPair("retstart", "0"),
            QueryPair("retmax", str(limit)),
            QueryPair("retmode", "json"),
            QueryPair("sort", "relevance"),
        )
        summary_pairs = (
            QueryPair("db", "pubmed"),
            QueryPair("retmode", "json"),
        )
        if identity_pubmed:
            identity_pairs = (
                QueryPair("tool", _NCBI_TOOL),
                QueryPair("email", _NCBI_EMAIL),
            )
            pairs += identity_pairs
            summary_pairs += identity_pairs
        return (
            _intent(route, OperationKind.SEARCH, route.policy.paths[0], pairs),
            _intent(
                route,
                OperationKind.CONDITIONAL_SUMMARY,
                route.policy.paths[1],
                summary_pairs,
                query_bindings=(
                    DeferredNumericCSVQueryBinding(
                        binding_id="pubmed_esearch_ids",
                        query_name="id",
                        max_items=limit,
                        max_item_chars=16,
                    ),
                ),
            ),
        )
    if _has_pubmed_identity_component(route):
        raise _planning_error(f"invalid_pubmed_route_identity:{route.route_id}")

    pairs_by_backend = {
        "arxiv_api": (
            QueryPair("search_query", f"all:{normalized_query}"),
            QueryPair("start", "0"),
            QueryPair("max_results", str(limit)),
        ),
        "semantic_scholar_graph_api": (
            QueryPair("query", normalized_query),
            QueryPair("offset", "0"),
            QueryPair("limit", str(limit)),
            QueryPair(
                "fields",
                "paperId,title,authors,abstract,tldr,externalIds,url,openAccessPdf",
            ),
        ),
        "crossref_api": (
            QueryPair("query", normalized_query),
            QueryPair("offset", "0"),
            QueryPair("rows", str(limit)),
            QueryPair("select", "DOI,title,author,abstract,URL,link"),
        ),
        "zenodo_records_api": (
            QueryPair("q", normalized_query),
            QueryPair("page", "1"),
            QueryPair("size", str(limit)),
        ),
        "figshare_public_api": (),
        "osf_api": (
            QueryPair("filter[title]", normalized_query),
            QueryPair("page", "1"),
            QueryPair("page[size]", str(limit)),
        ),
    }
    pairs = pairs_by_backend.get(route.backend_id)
    if pairs is None:
        query_key = "query" if "query" in route.policy.allowed_query_keys else route.policy.allowed_query_keys[0]
        pairs = (QueryPair(query_key, normalized_query),)
        if "limit" in route.policy.allowed_query_keys:
            pairs += (QueryPair("limit", str(limit)),)
    json_body_pairs = (
        (
            JSONBodyPair("search_for", normalized_query),
            JSONBodyPair("page", 1),
            JSONBodyPair("page_size", limit),
        )
        if route.backend_id == "figshare_public_api"
        else ()
    )
    return (
        _intent(
            route,
            OperationKind.SEARCH,
            route.policy.paths[0],
            pairs,
            json_body_pairs=json_body_pairs,
        ),
    )


def _intent(
    route: AccessRoute,
    operation_kind: OperationKind,
    path: str,
    query_pairs: tuple[QueryPair, ...],
    *,
    json_body_pairs: tuple[JSONBodyPair, ...] = (),
    query_bindings: tuple[DeferredNumericCSVQueryBinding, ...] = (),
) -> DispatchIntent:
    allowed = set(route.policy.allowed_query_keys)
    if any(pair.name not in allowed for pair in query_pairs) or any(
        binding.query_name not in allowed for binding in query_bindings
    ):
        raise _planning_error(f"intent_query_not_allowed:{route.route_id}")
    allowed_body = set(route.policy.allowed_json_body_keys)
    if any(pair.name not in allowed_body for pair in json_body_pairs):
        raise _planning_error(f"intent_json_body_not_allowed:{route.route_id}")
    if json_body_pairs and route.policy.methods[0] != "POST":
        raise _planning_error(f"intent_json_body_requires_post:{route.route_id}")
    return DispatchIntent(
        route_id=route.route_id,
        policy_digest=route.policy.policy_digest,
        operation_kind=operation_kind,
        method=route.policy.methods[0],
        path=path,
        query_pairs=query_pairs,
        limits=route.policy.limits,
        json_body_pairs=json_body_pairs,
        query_bindings=query_bindings,
    )


def _coalesce(logical: list[_LogicalAttempt]) -> tuple[PlannedDispatchGroup, ...]:
    grouped: dict[tuple[object, ...], list[_LogicalAttempt]] = {}
    for item in logical:
        key = (
            item.route.route_id,
            item.route.backend_id,
            item.route.adapter_id,
            item.route.adapter_version,
            item.normalized_query,
            item.filters,
            item.route.policy.policy_digest,
            item.route.fallback_order,
            item.intents,
            item.allowance,
        )
        grouped.setdefault(key, []).append(item)

    dispatch_groups: list[PlannedDispatchGroup] = []
    for group in grouped.values():
        first = group[0]
        dispatch_group_id = _dispatch_group_id(first)
        logical_attempts = tuple(
            PlannedLogicalAttempt(
                logical_attempt_id=_logical_attempt_id(item, dispatch_group_id),
                catalog_source_id=item.catalog_source_id,
                selection_reason=item.selection_reason,
                source_predicate=item.source_predicate,
            )
            for item in sorted(group, key=lambda item: (item.source_priority, item.catalog_source_id))
        )
        dispatch_groups.append(
            PlannedDispatchGroup(
                dispatch_group_id=dispatch_group_id,
                route_id=first.route.route_id,
                backend_id=first.route.backend_id,
                adapter_id=first.route.adapter_id,
                adapter_version=first.route.adapter_version,
                policy_digest=first.route.policy.policy_digest,
                limits=first.route.policy.limits,
                normalized_query=first.normalized_query,
                filters=first.filters,
                logical_attempts=logical_attempts,
                fallback_order=first.route.fallback_order,
                intents=first.intents,
                allowance=first.allowance,
            )
        )
    return tuple(dispatch_groups)


def _dispatch_group_id(logical: _LogicalAttempt) -> str:
    return _dispatch_group_id_from_parts(
        adapter_id=logical.route.adapter_id,
        adapter_version=logical.route.adapter_version,
        allowance=logical.allowance,
        backend_id=logical.route.backend_id,
        fallback_order=logical.route.fallback_order,
        filters=logical.filters,
        intents=logical.intents,
        limits=logical.route.policy.limits,
        normalized_query=logical.normalized_query,
        policy_digest=logical.route.policy.policy_digest,
        route_id=logical.route.route_id,
    )


def expected_dispatch_group_id(group: PlannedDispatchGroup) -> str:
    """Recompute one typed group's deterministic physical-work identity."""
    if type(group) is not PlannedDispatchGroup:
        raise TypeError("group_must_be_planned_dispatch_group")
    return _dispatch_group_id_from_parts(
        adapter_id=group.adapter_id,
        adapter_version=group.adapter_version,
        allowance=group.allowance,
        backend_id=group.backend_id,
        fallback_order=group.fallback_order,
        filters=group.filters,
        intents=group.intents,
        limits=group.limits,
        normalized_query=group.normalized_query,
        policy_digest=group.policy_digest,
        route_id=group.route_id,
    )


def _dispatch_group_id_from_parts(
    *,
    adapter_id: str,
    adapter_version: str,
    allowance: DispatchAllowance,
    backend_id: str,
    fallback_order: int,
    filters: tuple[QueryPair, ...],
    intents: tuple[DispatchIntent, ...],
    limits: RouteLimits,
    normalized_query: str,
    policy_digest: str,
    route_id: str,
) -> str:
    """Hash the canonical physical-work material used by planner and executor."""
    material = {
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "allowance": asdict(allowance),
        "backend_id": backend_id,
        "fallback_order": fallback_order,
        "filters": [asdict(item) for item in filters],
        "intents": [asdict(item) for item in intents],
        "limits": asdict(limits),
        "normalized_query": normalized_query,
        "policy_digest": policy_digest,
        "route_id": route_id,
    }
    return f"dispatch_group_v2_{hashlib.sha256(_canonical_json(material)).hexdigest()[:24]}"


def _logical_attempt_id(logical: _LogicalAttempt, dispatch_group_id: str) -> str:
    return _logical_attempt_id_from_parts(
        catalog_source_id=logical.catalog_source_id,
        dispatch_group_id=dispatch_group_id,
        selection_reason=logical.selection_reason,
        source_predicate=logical.source_predicate,
    )


def expected_logical_attempt_id(attempt: PlannedLogicalAttempt, expected_group_id: str) -> str:
    """Recompute one typed logical attempt's deterministic attribution identity."""
    if type(attempt) is not PlannedLogicalAttempt:
        raise TypeError("attempt_must_be_planned_logical_attempt")
    if type(expected_group_id) is not str or not expected_group_id:
        raise TypeError("expected_group_id_must_be_nonempty_string")
    return _logical_attempt_id_from_parts(
        catalog_source_id=attempt.catalog_source_id,
        dispatch_group_id=expected_group_id,
        selection_reason=attempt.selection_reason,
        source_predicate=attempt.source_predicate,
    )


def _logical_attempt_id_from_parts(
    *,
    catalog_source_id: str,
    dispatch_group_id: str,
    selection_reason: str,
    source_predicate: SourcePredicate | None,
) -> str:
    """Hash the canonical logical-attribution material used at both boundaries."""
    material = {
        "catalog_source_id": catalog_source_id,
        "dispatch_group_id": dispatch_group_id,
        "selection_reason": selection_reason,
        "source_predicate": asdict(source_predicate) if source_predicate is not None else None,
    }
    return f"logical_attempt_v2_{hashlib.sha256(_canonical_json(material)).hexdigest()[:24]}"


def _enforce_budget(
    allowance: PlannedBudgetAllowance,
    budget: BudgetCeilings,
) -> None:
    violation = budget_ceiling_violation(allowance, budget)
    if violation is not None:
        raise _planning_error(f"budget_exceeded:{violation}")


def _validate_readiness_references(
    registry: DiscoveryRegistry,
    readiness: ReadinessOverlay,
) -> None:
    known_route_ids = {route.route_id for route in registry.routes}
    for entry in readiness.routes:
        if entry.route_id not in known_route_ids:
            raise _planning_error(f"unknown_readiness_route:{entry.route_id}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
