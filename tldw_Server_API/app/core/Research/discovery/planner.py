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
    ExactQueryValuePolicy,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
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


class PlanningError(ValueError):
    """Typed pure-planning failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


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
                raise PlanningError(f"missing_readiness:{route.route_id}")
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
            raise PlanningError(f"unknown_source:{source_id}") from None
        resolved[source.catalog_source_id] = source
    return tuple(sorted(resolved.values(), key=lambda source: (source.priority, source.catalog_source_id)))


def _normalize_planning_query(query: PlanningQuery) -> _NormalizedPlanningQuery:
    """Validate one exact query type and return its route-selection values."""
    if type(query) is str:
        if not query.strip():
            raise PlanningError("query_must_be_nonempty")
        return _NormalizedPlanningQuery(QueryMode.STRUCTURED_QUERY, _normalize_query(query))
    if type(query) is GeneralFreeTextQuery:
        if type(query.text) is not str:
            raise PlanningError("general_query_text_must_be_exact_string")
        if len(query.text) > _GENERAL_MAX_RAW_CHARS:
            raise PlanningError("general_query_input_limit_exceeded")
        try:
            raw_utf8_bytes = len(query.text.encode("utf-8"))
        except UnicodeEncodeError:
            raise PlanningError("general_query_contains_invalid_unicode") from None
        if raw_utf8_bytes > _GENERAL_MAX_RAW_UTF8_BYTES:
            raise PlanningError("general_query_input_limit_exceeded")
        if any(unicodedata.category(character).startswith("C") for character in query.text):
            raise PlanningError("general_query_contains_invalid_unicode")
        canonical = unicodedata.normalize("NFKC", query.text)
        terms = _unicode_alphanumeric_terms(canonical)
        if not terms:
            raise PlanningError("general_query_requires_term")
        return _NormalizedPlanningQuery(
            QueryMode.GENERAL_FREE_TEXT,
            " ".join(terms),
            terms=terms,
        )
    if type(query) is IdentifierLookupQuery:
        if type(query.doi) is not str:
            raise PlanningError("doi_must_be_exact_string")
        if not query.doi.isascii() or query.doi.count("/") != 1:
            raise PlanningError("invalid_doi")
        registrant, suffix = query.doi.split("/", 1)
        if (
            _DOI_REGISTRANT.fullmatch(registrant) is None
            or not suffix
            or len(suffix) > 128
            or not suffix[0].isalnum()
            or any(character in " /\\%?#" or not "!" <= character <= "~" for character in suffix)
        ):
            raise PlanningError("invalid_doi")
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
            raise PlanningError("invalid_date_interval")
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
                raise PlanningError("invalid_category")
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
                raise PlanningError("general_query_term_limit_exceeded")
            if len(current) == _GENERAL_MAX_TERM_CHARS:
                raise PlanningError("general_query_term_limit_exceeded")
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
        raise PlanningError("invalid_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise PlanningError("invalid_date") from None
    if parsed.isoformat() != value:
        raise PlanningError("invalid_date")
    return parsed


def _normalize_query(query: str) -> str:
    try:
        query.encode("utf-8")
    except UnicodeEncodeError:
        raise PlanningError("query_contains_invalid_unicode") from None
    return " ".join(unicodedata.normalize("NFKC", query).split()).casefold()


def _build_typed_intents(
    route: AccessRoute,
    query: _NormalizedPlanningQuery,
    result_limit: int,
) -> tuple[DispatchIntent, ...]:
    """Render typed query values only through closed route policy shapes."""
    if route.policy.allowed_json_body_keys:
        raise PlanningError(f"typed_intent_json_body_not_supported:{route.route_id}")
    if query.mode is QueryMode.GENERAL_FREE_TEXT:
        if route.policy.path_template is not None or len(route.policy.paths) != 1:
            raise PlanningError(f"invalid_general_query_path_policy:{route.route_id}")
        policies = {policy.name: policy for policy in route.policy.query_value_policies}
        pairs: list[QueryPair] = []
        literal_terms_seen = False
        for name in route.policy.allowed_query_keys:
            policy = policies.get(name)
            if type(policy) is LiteralTermsQueryValuePolicy:
                if (
                    literal_terms_seen
                    or name != "query"
                    or not 1 <= len(query.terms) <= policy.max_terms
                    or any(len(term) > policy.max_term_chars for term in query.terms)
                ):
                    raise PlanningError(f"invalid_literal_terms_policy:{route.route_id}")
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
                limit = min(result_limit, route.policy.limits.max_results)
                if limit > policy.maximum:
                    raise PlanningError(f"typed_result_limit_exceeds_policy:{route.route_id}")
                pairs.append(QueryPair(name, str(limit)))
            else:
                raise PlanningError(f"invalid_general_query_value_policy:{route.route_id}")
        if not literal_terms_seen or len(pairs) != len(route.policy.allowed_query_keys):
            raise PlanningError(f"incomplete_general_query_policy:{route.route_id}")
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
        raise PlanningError(f"typed_query_requires_path_template:{route.route_id}")
    if query.mode is QueryMode.IDENTIFIER_LOOKUP:
        if (
            query.doi_registrant is None
            or query.doi_suffix is None
            or template.pagination_segment_index is not None
            or route.policy.allowed_query_keys
            or route.policy.query_value_policies
        ):
            raise PlanningError(f"invalid_identifier_route_policy:{route.route_id}")
        path = _render_path_template(
            template,
            {
                PathSlotKind.DOI_REGISTRANT: (query.doi_registrant,),
                PathSlotKind.DOI_SUFFIX: (query.doi_suffix,),
            },
        )
        return (_intent(route, OperationKind.SEARCH, path, ()),)

    if query.mode not in {QueryMode.DATE_INTERVAL, QueryMode.CATEGORY_BROWSE}:
        raise PlanningError(f"unsupported_typed_query_mode:{route.route_id}")
    if query.start_date is None or query.end_date is None:
        raise PlanningError(f"invalid_interval_query:{route.route_id}")
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
        raise PlanningError(f"invalid_interval_query_policy:{route.route_id}")
    category_policy = policies[0]
    if query.category is None:
        if category_policy.required:
            raise PlanningError(f"interval_category_required:{route.route_id}")
        pairs = ()
    else:
        if len(query.category) > category_policy.max_chars:
            raise PlanningError(f"interval_category_exceeds_policy:{route.route_id}")
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
            raise PlanningError("path_template_shape_mismatch")
        value = remaining[segment.kind].pop(0)
        if not value.isascii() or len(value) > segment.max_chars:
            raise PlanningError("path_template_value_invalid")
        rendered.append(_encode_ascii_path_segment(value))
    if any(items for items in remaining.values()):
        raise PlanningError("path_template_shape_mismatch")
    return f"/{'/'.join(rendered)}"


def _encode_ascii_path_segment(value: str) -> str:
    """Percent-encode one validated ASCII segment without transport imports."""
    return "".join(character if character in _RFC3986_UNRESERVED else f"%{ord(character):02X}" for character in value)


def _build_intents(
    route: AccessRoute,
    normalized_query: str,
    result_limit: int,
) -> tuple[DispatchIntent, ...]:
    limit = min(result_limit, route.policy.limits.max_results)
    if route.backend_id == "ncbi_eutils_pubmed":
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
        raise PlanningError(f"intent_query_not_allowed:{route.route_id}")
    allowed_body = set(route.policy.allowed_json_body_keys)
    if any(pair.name not in allowed_body for pair in json_body_pairs):
        raise PlanningError(f"intent_json_body_not_allowed:{route.route_id}")
    if json_body_pairs and route.policy.methods[0] != "POST":
        raise PlanningError(f"intent_json_body_requires_post:{route.route_id}")
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
        raise PlanningError(f"budget_exceeded:{violation}")


def _validate_readiness_references(
    registry: DiscoveryRegistry,
    readiness: ReadinessOverlay,
) -> None:
    known_route_ids = {route.route_id for route in registry.routes}
    for entry in readiness.routes:
        if entry.route_id not in known_route_ids:
            raise PlanningError(f"unknown_readiness_route:{entry.route_id}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
