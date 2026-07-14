"""Deterministic budget and coalescing tests for the pure V2 planner."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    AttributionMatch,
    BackendDefinition,
    BudgetCeilings,
    CredentialRequirement,
    CredentialStatus,
    ExactOrigin,
    ExecutionMode,
    PredicateOperator,
    QueryMode,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    RouteReadiness,
    SkippedCode,
    SkippedStatus,
    SourceConstraint,
    SourceDefinition,
    SourcePredicate,
    SourceRouteReference,
    evaluate_source_predicate,
)
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningError,
    PlanningRequest,
    canonical_plan_bytes,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)

_FOUNDATION_SOURCE_IDS = (
    "openalex",
    "semantic_scholar",
    "crossref",
    "arxiv",
    "pubmed",
    "zenodo",
    "figshare",
    "osf",
)


def _budget(**changes: int) -> BudgetCeilings:
    values = {
        "max_route_attempts": 16,
        "max_physical_dispatches": 20,
        "max_pages_per_route": 1,
        "max_redirects": 0,
        "max_retries": 0,
        "max_wall_time_ms": 500_000,
        "max_results": 100,
    }
    values.update(changes)
    return BudgetCeilings(**values)


def _request(
    source_ids: tuple[str, ...],
    *,
    query: str = "  Causal   Inference  ",
    filters: tuple[QueryPair, ...] = (),
    result_limit: int = 25,
) -> PlanningRequest:
    return PlanningRequest(
        source_ids=source_ids,
        query=query,
        filters=filters,
        result_limit=result_limit,
    )


def _aggregator_registry(
    *,
    second_predicate: SourcePredicate | None = None,
    limits: RouteLimits | None = None,
    max_physical_dispatches: int = 1,
) -> tuple[DiscoveryRegistry, ReadinessOverlay]:
    first_predicate = SourcePredicate(
        field_path=("source", "collection"),
        operator=PredicateOperator.EQUALS_ANY,
        values=("shared-index",),
    )
    policy = RoutePolicy(
        policy_version="synthetic-policy-v1",
        origin=ExactOrigin("https", "aggregator.example.test", 443),
        methods=("GET",),
        paths=("/search",),
        allowed_query_keys=("query", "limit"),
        limits=limits or RouteLimits(1, 0, 0, 2_000, 16_384, 50),
    )
    route = AccessRoute(
        route_id="shared_aggregator_search",
        backend_id="shared_aggregator",
        adapter_id="shared_aggregator_v2",
        route_kind=RouteKind.AGGREGATOR,
        query_modes=(QueryMode.STRUCTURED_QUERY,),
        source_constraint=SourceConstraint.PROVIDER_SOURCE_FILTER,
        attribution_basis="provider_source_field",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=max_physical_dispatches,
        adapter_version="synthetic-v1",
        policy=policy,
    )
    sources = (
        SourceDefinition(
            catalog_source_id="target_a",
            display_name="Target A",
            aliases=(),
            categories=("synthetic",),
            content_types=("works",),
            surfaces=("standalone_search",),
            route_references=(SourceRouteReference(route.route_id, first_predicate),),
            site_hosts=("target-a.example.test",),
            priority=10,
            catalog_version="synthetic-catalog-v1",
        ),
        SourceDefinition(
            catalog_source_id="target_b",
            display_name="Target B",
            aliases=(),
            categories=("synthetic",),
            content_types=("works",),
            surfaces=("standalone_search",),
            route_references=(SourceRouteReference(route.route_id, second_predicate or first_predicate),),
            site_hosts=("target-b.example.test",),
            priority=20,
            catalog_version="synthetic-catalog-v1",
        ),
    )
    registry = DiscoveryRegistry(
        catalog_version="synthetic-catalog-v1",
        registry_version="synthetic-registry-v1",
        sources=sources,
        routes=(route,),
        backends=(BackendDefinition("shared_aggregator", "Shared Aggregator"),),
    )
    readiness = ReadinessOverlay(
        overlay_version="synthetic-readiness-v1",
        execution_mode=ExecutionMode.SYNTHETIC,
        routes=(
            RouteReadiness(
                route_id=route.route_id,
                state=ReadinessState.READY,
                credential_status=CredentialStatus.NOT_REQUIRED,
                reason="synthetic_ready",
            ),
        ),
    )
    return registry, readiness


def test_openalex_is_typed_unavailable_with_no_attempt_or_dispatch_allowance() -> None:
    plan = compile_discovery_plan(
        _request(("openalex",)),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_budget(),
    )

    assert plan.attempts == ()
    assert plan.allowance.route_attempts == 0
    assert plan.allowance.physical_dispatches == 0
    assert plan.allowance.returned_results == 0
    assert len(plan.skipped) == 1
    assert plan.skipped[0].status is SkippedStatus.UNAVAILABLE
    assert plan.skipped[0].code is SkippedCode.CREDENTIALED_OUT_OF_SCOPE
    assert plan.skipped[0].requested_source_id == "openalex"

    zero_result_budget_plan = compile_discovery_plan(
        _request(("openalex",)),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_budget(max_results=0),
    )
    assert zero_result_budget_plan.allowance.returned_results == 0


def test_openalex_has_no_positive_branch_even_if_readiness_is_marked_ready() -> None:
    registry = foundation_registry()
    overlay = foundation_readiness(ExecutionMode.SYNTHETIC)
    openalex_route_id = registry.get_source("openalex").route_references[0].route_id
    optimistic_overlay = replace(
        overlay,
        routes=tuple(
            (
                replace(
                    entry,
                    state=ReadinessState.READY,
                    credential_status=CredentialStatus.NOT_REQUIRED,
                    reason="must_not_enable_credentialed_route",
                )
                if entry.route_id == openalex_route_id
                else entry
            )
            for entry in overlay.routes
        ),
    )

    plan = compile_discovery_plan(
        _request(("openalex",)),
        registry=registry,
        readiness=optimistic_overlay,
        budget=_budget(),
    )

    assert plan.attempts == ()
    assert plan.allowance.physical_dispatches == 0
    assert plan.skipped[0].code is SkippedCode.CREDENTIALED_OUT_OF_SCOPE


def test_foundation_plan_compiles_seven_routes_and_pubmed_two_dispatch_allowance() -> None:
    plan = compile_discovery_plan(
        _request(_FOUNDATION_SOURCE_IDS),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_budget(),
    )

    assert [attempt.backend_id for attempt in plan.attempts] == [
        "semantic_scholar_graph_api",
        "crossref_api",
        "arxiv_api",
        "ncbi_eutils_pubmed",
        "zenodo_records_api",
        "figshare_public_api",
        "osf_api",
    ]
    assert plan.allowance.route_attempts == 7
    assert plan.allowance.physical_dispatches == 8
    assert plan.allowance.max_pages_per_route == 1
    assert plan.allowance.redirects == 0
    assert plan.allowance.retries == 0
    assert plan.allowance.returned_results == 25
    assert [skipped.requested_source_id for skipped in plan.skipped] == ["openalex"]


def test_pubmed_declares_two_possible_dispatches_without_runtime_accounting() -> None:
    plan = compile_discovery_plan(
        _request(("pubmed",)),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    attempt = plan.attempts[0]

    assert attempt.allowance.physical_dispatches == 2
    assert [intent.operation_kind.value for intent in attempt.intents] == [
        "search",
        "conditional_summary",
    ]
    assert [intent.path for intent in attempt.intents] == [
        "/entrez/eutils/esearch.fcgi",
        "/entrez/eutils/esummary.fcgi",
    ]
    assert not any(hasattr(attempt.allowance, name) for name in ("reservation", "reserved", "debit", "release"))


def test_plan_bytes_and_attempt_ids_are_deterministic_after_input_normalization() -> None:
    registry = foundation_registry()
    readiness = foundation_readiness(ExecutionMode.OFFLINE_FIXTURE)
    first = compile_discovery_plan(
        _request(
            ("crossref_metadata_search", "arxiv", "arxiv"),
            filters=(QueryPair("year", "2025"), QueryPair("venue", "JMLR")),
        ),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    second = compile_discovery_plan(
        _request(
            ("arxiv", "crossref"),
            query="causal inference",
            filters=(QueryPair("venue", "JMLR"), QueryPair("year", "2025")),
        ),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )

    assert canonical_plan_bytes(first) == canonical_plan_bytes(second)
    assert tuple(attempt.attempt_id for attempt in first.attempts) == tuple(
        attempt.attempt_id for attempt in second.attempts
    )


def test_shared_backend_work_coalesces_distinct_attribution_predicates() -> None:
    second_predicate = SourcePredicate(
        field_path=("source", "collection"),
        operator=PredicateOperator.EQUALS_ANY,
        values=("different-index",),
    )
    registry, readiness = _aggregator_registry(second_predicate=second_predicate)
    plan = compile_discovery_plan(
        _request(("target_b", "target_a")),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )

    assert plan.allowance.route_attempts == 2
    assert plan.allowance.physical_dispatches == 1
    assert len(plan.attempts) == 1
    assert [target.catalog_source_id for target in plan.attempts[0].requested_targets] == [
        "target_a",
        "target_b",
    ]
    assert plan.attempts[0].backend_id == "shared_aggregator"
    assert [target.source_predicate for target in plan.attempts[0].requested_targets] == [
        registry.get_source("target_a").route_references[0].source_predicate,
        second_predicate,
    ]


def test_shared_aggregator_predicates_keep_three_valued_target_attribution() -> None:
    second_predicate = SourcePredicate(
        field_path=("source", "collection"),
        operator=PredicateOperator.EQUALS_ANY,
        values=("different-index",),
    )
    registry, readiness = _aggregator_registry(second_predicate=second_predicate)
    plan = compile_discovery_plan(
        _request(("target_a", "target_b")),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    targets = plan.attempts[0].requested_targets

    first_predicate = targets[0].source_predicate
    second_predicate = targets[1].source_predicate
    assert first_predicate is not None
    assert second_predicate is not None
    record = {"source": {"collection": "shared-index"}}
    assert evaluate_source_predicate(first_predicate, record) is AttributionMatch.MATCH
    assert evaluate_source_predicate(second_predicate, record) is AttributionMatch.NON_MATCH
    assert evaluate_source_predicate(first_predicate, {"source": {}}) is AttributionMatch.AMBIGUOUS
    assert evaluate_source_predicate(second_predicate, {"source": {}}) is AttributionMatch.AMBIGUOUS


def test_physical_and_wall_time_allowances_cover_pages_redirects_and_retries() -> None:
    limits = RouteLimits(
        max_pages=2,
        max_redirects=1,
        max_retries=1,
        timeout_ms=2_000,
        max_response_bytes=16_384,
        max_results=50,
    )
    registry, readiness = _aggregator_registry(
        limits=limits,
        max_physical_dispatches=4,
    )

    plan = compile_discovery_plan(
        _request(("target_a",)),
        registry=registry,
        readiness=readiness,
        budget=_budget(
            max_physical_dispatches=4,
            max_pages_per_route=2,
            max_redirects=1,
            max_retries=1,
            max_wall_time_ms=8_000,
        ),
    )

    assert plan.attempts[0].allowance.physical_dispatches == 4
    assert plan.allowance.physical_dispatches == 4
    assert plan.allowance.max_pages_per_route == 2
    assert plan.allowance.redirects == 1
    assert plan.allowance.retries == 1
    assert plan.allowance.aggregate_wall_time_ms == 8_000

    for changes, code in (
        ({"max_physical_dispatches": 3}, "budget_exceeded:physical_dispatches"),
        ({"max_pages_per_route": 1}, "budget_exceeded:pages_per_route"),
        ({"max_redirects": 0}, "budget_exceeded:redirects"),
        ({"max_retries": 0}, "budget_exceeded:retries"),
        ({"max_wall_time_ms": 7_999}, "budget_exceeded:wall_time_ms"),
    ):
        with pytest.raises(PlanningError, match=code):
            compile_discovery_plan(
                _request(("target_a",)),
                registry=registry,
                readiness=readiness,
                budget=_budget(
                    **{
                        "max_physical_dispatches": 4,
                        "max_pages_per_route": 2,
                        "max_redirects": 1,
                        "max_retries": 1,
                        "max_wall_time_ms": 8_000,
                        **changes,
                    }
                ),
            )


def test_planner_preserves_ordered_fallback_attempts() -> None:
    registry = foundation_registry()
    source = registry.get_source("arxiv")
    primary = registry.get_route(source.route_references[0].route_id)
    fallback_policy = replace(
        primary.policy,
        paths=("/api/fallback-query",),
        policy_digest="",
    )
    fallback = replace(
        primary,
        route_id="arxiv_fixture_fallback",
        fallback_order=1,
        policy=fallback_policy,
    )
    source = replace(
        source,
        route_references=(
            SourceRouteReference(primary.route_id, None),
            SourceRouteReference(fallback.route_id, None),
        ),
    )
    synthetic_registry = DiscoveryRegistry(
        catalog_version=registry.catalog_version,
        registry_version="fallback-registry-v1",
        sources=(source,),
        routes=(fallback, primary),
        backends=(registry.get_backend(primary.backend_id),),
    )
    readiness = ReadinessOverlay(
        overlay_version="fallback-readiness-v1",
        execution_mode=ExecutionMode.SYNTHETIC,
        routes=(
            RouteReadiness(primary.route_id, ReadinessState.READY, CredentialStatus.NOT_REQUIRED, "ready"),
            RouteReadiness(fallback.route_id, ReadinessState.READY, CredentialStatus.NOT_REQUIRED, "ready"),
        ),
    )

    plan = compile_discovery_plan(
        _request(("arxiv",)),
        registry=synthetic_registry,
        readiness=readiness,
        budget=_budget(),
    )

    assert [(attempt.route_id, attempt.fallback_order) for attempt in plan.attempts] == [
        (primary.route_id, 0),
        (fallback.route_id, 1),
    ]


@pytest.mark.parametrize(
    ("budget", "code"),
    [
        (_budget(max_route_attempts=6), "budget_exceeded:route_attempts"),
        (_budget(max_physical_dispatches=7), "budget_exceeded:physical_dispatches"),
        (_budget(max_pages_per_route=0), "budget_exceeded:pages_per_route"),
        (_budget(max_wall_time_ms=1), "budget_exceeded:wall_time_ms"),
        (_budget(max_results=24), "budget_exceeded:returned_results"),
    ],
)
def test_planner_rejects_impossible_budget_dimensions(
    budget: BudgetCeilings,
    code: str,
) -> None:
    with pytest.raises(PlanningError) as exc_info:
        compile_discovery_plan(
            _request(_FOUNDATION_SOURCE_IDS),
            registry=foundation_registry(),
            readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
            budget=budget,
        )

    assert exc_info.value.code == code


def test_planner_requires_complete_immutable_readiness() -> None:
    registry = foundation_registry()
    with pytest.raises(PlanningError) as exc_info:
        compile_discovery_plan(
            _request(("arxiv",)),
            registry=registry,
            readiness=ReadinessOverlay(
                overlay_version="empty-overlay-v1",
                execution_mode=ExecutionMode.SYNTHETIC,
                routes=(),
            ),
            budget=_budget(),
        )

    assert exc_info.value.code == "missing_readiness:arxiv_arxiv_api_direct"


@settings(max_examples=35, deadline=None)
@given(
    source_ids=st.lists(
        st.sampled_from(_FOUNDATION_SOURCE_IDS),
        min_size=1,
        max_size=len(_FOUNDATION_SOURCE_IDS),
        unique=True,
    ),
    query=st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=1,
        max_size=40,
    ).filter(lambda value: bool(value.strip())),
    result_limit=st.integers(min_value=1, max_value=100),
)
def test_plan_property_invariants(
    source_ids: list[str],
    query: str,
    result_limit: int,
) -> None:
    registry = foundation_registry()
    readiness = foundation_readiness(ExecutionMode.SYNTHETIC)
    budget = _budget()
    request = _request(tuple(source_ids), query=query, result_limit=result_limit)
    first = compile_discovery_plan(request, registry=registry, readiness=readiness, budget=budget)
    second = compile_discovery_plan(request, registry=registry, readiness=readiness, budget=budget)
    allowance = first.allowance

    assert canonical_plan_bytes(first) == canonical_plan_bytes(second)
    assert all(
        value >= 0
        for value in (
            allowance.route_attempts,
            allowance.physical_dispatches,
            allowance.max_pages_per_route,
            allowance.redirects,
            allowance.retries,
            allowance.aggregate_wall_time_ms,
            allowance.returned_results,
        )
    )
    assert allowance.route_attempts <= budget.max_route_attempts
    assert allowance.physical_dispatches <= budget.max_physical_dispatches
    assert allowance.max_pages_per_route <= budget.max_pages_per_route
    assert allowance.redirects <= budget.max_redirects
    assert allowance.retries <= budget.max_retries
    assert allowance.aggregate_wall_time_ms <= budget.max_wall_time_ms
    assert allowance.returned_results <= budget.max_results
