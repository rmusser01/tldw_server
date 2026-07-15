"""Pure deterministic compiler for research discovery V2 plans."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass

from .contracts import (
    CREDENTIALED_ROUTE_SKIP_REASON,
    AccessRoute,
    BudgetCeilings,
    CredentialRequirement,
    DeferredNumericCSVQueryBinding,
    DiscoveryPlan,
    DispatchAllowance,
    DispatchIntent,
    JSONBodyPair,
    OperationKind,
    PlannedBudgetAllowance,
    PlannedDispatchGroup,
    PlannedLogicalAttempt,
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


class PlanningError(ValueError):
    """Typed pure-planning failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Explicit V2 selection and normalized planning inputs."""

    source_ids: tuple[str, ...]
    query: str
    filters: tuple[QueryPair, ...]
    result_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_ids, tuple):
            raise TypeError("source_ids_must_be_tuple")
        if not self.source_ids or any(
            not isinstance(source_id, str) or not source_id.strip() for source_id in self.source_ids
        ):
            raise ValueError("explicit_selection_requires_source_ids")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query_must_be_nonempty")
        if not isinstance(self.filters, tuple) or any(not isinstance(item, QueryPair) for item in self.filters):
            raise TypeError("filters_must_be_query_pair_tuple")
        if not isinstance(self.result_limit, int) or isinstance(self.result_limit, bool) or self.result_limit <= 0:
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

    normalized_query = _normalize_query(request.query)
    filters = tuple(sorted(request.filters, key=lambda item: (item.name, item.value)))
    sources = _resolve_sources(request.source_ids, registry)
    logical: list[_LogicalAttempt] = []
    skipped: list[SkippedTarget] = []

    for source in sources:
        references = {reference.route_id: reference for reference in source.route_references}
        for route in registry.routes_for_source(source.catalog_source_id):
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

            intents = _build_intents(route, normalized_query, request.result_limit)
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


def _normalize_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).split()).casefold()


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
