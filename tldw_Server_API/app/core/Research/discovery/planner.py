"""Pure deterministic compiler for research discovery V2 plans."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass

from .contracts import (
    AccessRoute,
    BudgetCeilings,
    CredentialRequirement,
    DiscoveryPlan,
    DispatchAllowance,
    DispatchIntent,
    OperationKind,
    PlannedAttempt,
    PlannedBudgetAllowance,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RequestedTarget,
    SkippedCode,
    SkippedStatus,
    SkippedTarget,
    SourceDefinition,
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
    target: RequestedTarget
    normalized_query: str
    filters: tuple[QueryPair, ...]
    intents: tuple[DispatchIntent, ...]
    allowance: DispatchAllowance


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
                        reason="credentialed_route_not_authorized_for_foundation",
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
                    target=RequestedTarget(
                        catalog_source_id=source.catalog_source_id,
                        selection_reason="explicit",
                        source_predicate=references[route.route_id].source_predicate,
                    ),
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
    attempts = _coalesce(logical)
    allowance = _plan_allowance(logical, attempts, request.result_limit, registry)
    _enforce_budget(allowance, budget)
    return DiscoveryPlan(
        planner_version=PLANNER_VERSION,
        catalog_version=registry.catalog_version,
        registry_version=registry.registry_version,
        readiness_version=readiness.overlay_version,
        execution_mode=readiness.execution_mode,
        normalized_query=normalized_query,
        filters=filters,
        attempts=attempts,
        skipped=tuple(skipped),
        ceilings=budget,
        allowance=allowance,
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
            QueryPair("id", "{esearch_ids}"),
            QueryPair("retmode", "json"),
        )
        return (
            _intent(route, OperationKind.SEARCH, route.policy.paths[0], pairs),
            _intent(
                route,
                OperationKind.CONDITIONAL_SUMMARY,
                route.policy.paths[1],
                summary_pairs,
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
        ),
        "crossref_api": (
            QueryPair("query", normalized_query),
            QueryPair("offset", "0"),
            QueryPair("rows", str(limit)),
        ),
        "zenodo_records_api": (
            QueryPair("q", normalized_query),
            QueryPair("page", "1"),
            QueryPair("size", str(limit)),
        ),
        "figshare_public_api": (
            QueryPair("search_for", normalized_query),
            QueryPair("page", "1"),
            QueryPair("page_size", str(limit)),
        ),
        "osf_api": (
            QueryPair("q", normalized_query),
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
    return (_intent(route, OperationKind.SEARCH, route.policy.paths[0], pairs),)


def _intent(
    route: AccessRoute,
    operation_kind: OperationKind,
    path: str,
    query_pairs: tuple[QueryPair, ...],
) -> DispatchIntent:
    allowed = set(route.policy.allowed_query_keys)
    if any(pair.name not in allowed for pair in query_pairs):
        raise PlanningError(f"intent_query_not_allowed:{route.route_id}")
    return DispatchIntent(
        route_id=route.route_id,
        policy_digest=route.policy.policy_digest,
        operation_kind=operation_kind,
        method=route.policy.methods[0],
        path=path,
        query_pairs=query_pairs,
        limits=route.policy.limits,
    )


def _coalesce(logical: list[_LogicalAttempt]) -> tuple[PlannedAttempt, ...]:
    grouped: dict[tuple[object, ...], list[_LogicalAttempt]] = {}
    for item in logical:
        key = (
            item.route.route_id,
            item.route.backend_id,
            item.normalized_query,
            item.filters,
            item.route.policy.policy_digest,
            item.route.fallback_order,
            item.intents,
            item.allowance,
        )
        grouped.setdefault(key, []).append(item)

    attempts: list[PlannedAttempt] = []
    for group in grouped.values():
        first = group[0]
        targets = tuple(
            item.target
            for item in sorted(
                group,
                key=lambda item: (item.source_priority, item.target.catalog_source_id),
            )
        )
        attempt_id = _attempt_id(first, targets)
        attempts.append(
            PlannedAttempt(
                attempt_id=attempt_id,
                route_id=first.route.route_id,
                backend_id=first.route.backend_id,
                policy_digest=first.route.policy.policy_digest,
                normalized_query=first.normalized_query,
                filters=first.filters,
                requested_targets=targets,
                fallback_order=first.route.fallback_order,
                intents=first.intents,
                allowance=first.allowance,
            )
        )
    return tuple(attempts)


def _attempt_id(
    logical: _LogicalAttempt,
    targets: tuple[RequestedTarget, ...],
) -> str:
    material = {
        "backend_id": logical.route.backend_id,
        "fallback_order": logical.route.fallback_order,
        "filters": [asdict(item) for item in logical.filters],
        "intents": [asdict(item) for item in logical.intents],
        "normalized_query": logical.normalized_query,
        "policy_digest": logical.route.policy.policy_digest,
        "route_id": logical.route.route_id,
        "targets": [asdict(item) for item in targets],
    }
    return f"attempt_v2_{hashlib.sha256(_canonical_json(material)).hexdigest()[:24]}"


def _plan_allowance(
    logical: list[_LogicalAttempt],
    attempts: tuple[PlannedAttempt, ...],
    result_limit: int,
    registry: DiscoveryRegistry,
) -> PlannedBudgetAllowance:
    physical = sum(attempt.allowance.physical_dispatches for attempt in attempts)
    pages = max((attempt.allowance.pages for attempt in attempts), default=0)
    redirects = sum(attempt.allowance.redirects for attempt in attempts)
    retries = sum(attempt.allowance.retries for attempt in attempts)
    wall_time = sum(
        registry.get_route(attempt.route_id).policy.limits.timeout_ms * attempt.allowance.physical_dispatches
        for attempt in attempts
    )
    return PlannedBudgetAllowance(
        route_attempts=len(logical),
        physical_dispatches=physical,
        max_pages_per_route=pages,
        redirects=redirects,
        retries=retries,
        aggregate_wall_time_ms=wall_time,
        returned_results=result_limit if attempts else 0,
    )


def _enforce_budget(
    allowance: PlannedBudgetAllowance,
    budget: BudgetCeilings,
) -> None:
    checks = (
        ("route_attempts", allowance.route_attempts, budget.max_route_attempts),
        (
            "physical_dispatches",
            allowance.physical_dispatches,
            budget.max_physical_dispatches,
        ),
        (
            "pages_per_route",
            allowance.max_pages_per_route,
            budget.max_pages_per_route,
        ),
        ("redirects", allowance.redirects, budget.max_redirects),
        ("retries", allowance.retries, budget.max_retries),
        ("wall_time_ms", allowance.aggregate_wall_time_ms, budget.max_wall_time_ms),
        ("returned_results", allowance.returned_results, budget.max_results),
    )
    for name, planned, ceiling in checks:
        if planned > ceiling:
            raise PlanningError(f"budget_exceeded:{name}")


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
