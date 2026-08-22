"""Deterministic budget and coalescing tests for the pure V2 planner."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Research.discovery import planner as planner_module
from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    AttributionMatch,
    BackendDefinition,
    BoundedDecimalQueryValuePolicy,
    BoundedTextQueryValuePolicy,
    BudgetCeilings,
    CredentialRequirement,
    CredentialStatus,
    DeferredNumericCSVQueryBinding,
    ExactOrigin,
    ExactQueryValuePolicy,
    ExecutionMode,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
    OpaqueCursorQueryValuePolicy,
    PathSlot,
    PathSlotKind,
    PathTemplate,
    PlannedDispatchGroup,
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
    canonical_plan_digest,
    evaluate_source_predicate,
)
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningError,
    PlanningRequest,
    canonical_plan_bytes,
    compile_discovery_plan,
    expected_dispatch_group_id,
    expected_logical_attempt_id,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)

pytestmark = pytest.mark.unit

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
    query: object = "  Causal   Inference  ",
    filters: tuple[QueryPair, ...] = (),
    result_limit: int = 25,
) -> PlanningRequest:
    return PlanningRequest(
        source_ids=source_ids,
        query=query,
        filters=filters,
        result_limit=result_limit,
    )


def _typed_query_registry() -> tuple[DiscoveryRegistry, ReadinessOverlay]:
    limits = RouteLimits(1, 0, 0, 2_000, 16_384, 50)
    general_policy = RoutePolicy(
        policy_version="typed-policy-v1",
        origin=ExactOrigin("https", "www.ebi.ac.uk", 443),
        methods=("GET",),
        paths=("/europepmc/webservices/rest/search",),
        allowed_query_keys=("query", "format", "resultType", "pageSize"),
        limits=limits,
        query_value_policies=(
            LiteralTermsQueryValuePolicy(
                "query",
                ' AND SRC:PPR AND PUBLISHER:"bioRxiv"',
                16,
                64,
            ),
            ExactQueryValuePolicy("format", "json"),
            ExactQueryValuePolicy("resultType", "core"),
            BoundedDecimalQueryValuePolicy("pageSize", 50),
        ),
    )
    details_origin = ExactOrigin("https", "api.biorxiv.org", 443)
    doi_policy = RoutePolicy(
        policy_version="typed-policy-v1",
        origin=details_origin,
        methods=("GET",),
        paths=(),
        allowed_query_keys=(),
        limits=limits,
        path_template=PathTemplate(
            (
                "details",
                "biorxiv",
                PathSlot(PathSlotKind.DOI_REGISTRANT, 12),
                PathSlot(PathSlotKind.DOI_SUFFIX, 128),
                "na",
                "json",
            )
        ),
    )
    interval_limits = replace(limits, max_pages=2)
    interval_policy = RoutePolicy(
        policy_version="typed-policy-v1",
        origin=details_origin,
        methods=("GET",),
        paths=(),
        allowed_query_keys=("category",),
        limits=interval_limits,
        path_template=PathTemplate(
            (
                "details",
                "biorxiv",
                PathSlot(PathSlotKind.DATE, 10),
                PathSlot(PathSlotKind.DATE, 10),
                PathSlot(PathSlotKind.UINT, 10),
                "json",
            ),
            pagination_segment_index=4,
        ),
        query_value_policies=(BoundedTextQueryValuePolicy("category", 128),),
    )
    routes = (
        AccessRoute(
            route_id="typed_general_search",
            backend_id="typed_general_backend",
            adapter_id="typed_general_adapter",
            route_kind=RouteKind.AGGREGATOR,
            query_modes=(QueryMode.GENERAL_FREE_TEXT,),
            source_constraint=SourceConstraint.NATIVE_CORPUS,
            attribution_basis="native_response",
            credential_requirement=CredentialRequirement.NONE,
            fallback_order=0,
            max_physical_dispatches=1,
            adapter_version="typed-v1",
            policy=general_policy,
        ),
        AccessRoute(
            route_id="typed_identifier_lookup",
            backend_id="typed_details_backend",
            adapter_id="typed_details_adapter",
            route_kind=RouteKind.DIRECT,
            query_modes=(QueryMode.IDENTIFIER_LOOKUP,),
            source_constraint=SourceConstraint.NATIVE_CORPUS,
            attribution_basis="native_response",
            credential_requirement=CredentialRequirement.NONE,
            fallback_order=0,
            max_physical_dispatches=1,
            adapter_version="typed-v1",
            policy=doi_policy,
        ),
        AccessRoute(
            route_id="typed_interval_browse",
            backend_id="typed_details_backend",
            adapter_id="typed_details_adapter",
            route_kind=RouteKind.DIRECT,
            query_modes=(QueryMode.DATE_INTERVAL, QueryMode.CATEGORY_BROWSE),
            source_constraint=SourceConstraint.NATIVE_CORPUS,
            attribution_basis="native_response",
            credential_requirement=CredentialRequirement.NONE,
            fallback_order=0,
            max_physical_dispatches=2,
            adapter_version="typed-v1",
            policy=interval_policy,
        ),
    )
    source = SourceDefinition(
        catalog_source_id="typed_source",
        display_name="Typed Source",
        aliases=(),
        categories=("synthetic",),
        content_types=("works",),
        surfaces=("standalone_search",),
        route_references=tuple(SourceRouteReference(route.route_id, None) for route in routes),
        site_hosts=("typed.example.test",),
        priority=10,
        catalog_version="typed-catalog-v1",
    )
    registry = DiscoveryRegistry(
        catalog_version="typed-catalog-v1",
        registry_version="typed-registry-v1",
        sources=(source,),
        routes=routes,
        backends=(
            BackendDefinition("typed_general_backend", "Typed General Backend"),
            BackendDefinition("typed_details_backend", "Typed Details Backend"),
        ),
    )
    readiness = ReadinessOverlay(
        overlay_version="typed-readiness-v1",
        execution_mode=ExecutionMode.SYNTHETIC,
        routes=tuple(
            RouteReadiness(
                route.route_id,
                ReadinessState.READY,
                CredentialStatus.NOT_REQUIRED,
                "typed_ready",
            )
            for route in routes
        ),
    )
    return registry, readiness


def _opaque_query_registry() -> tuple[DiscoveryRegistry, ReadinessOverlay]:
    limits = RouteLimits(2, 0, 0, 250, 65_536, 100, 16_384)
    policy = RoutePolicy(
        policy_version="opaque-query-policy-v1",
        origin=ExactOrigin("https", "clinical.example.test", 443),
        methods=("GET",),
        paths=("/api/v2/studies",),
        allowed_query_keys=("query.term", "pageToken", "format", "pageSize"),
        limits=limits,
        pagination_query_key="pageToken",
        query_value_policies=(
            LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
            OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=False),
            ExactQueryValuePolicy("format", "json"),
            BoundedDecimalQueryValuePolicy("pageSize", 50),
        ),
    )
    route = AccessRoute(
        route_id="opaque_query_search",
        backend_id="opaque_query_backend",
        adapter_id="opaque_query_adapter",
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_response",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version="opaque-v1",
        policy=policy,
    )
    registry = DiscoveryRegistry(
        catalog_version="opaque-query-catalog-v1",
        registry_version="opaque-query-registry-v1",
        sources=(
            SourceDefinition(
                catalog_source_id="opaque_query_source",
                display_name="Opaque Query Source",
                aliases=(),
                categories=("synthetic",),
                content_types=("records",),
                surfaces=("standalone_search",),
                route_references=(SourceRouteReference(route.route_id, None),),
                site_hosts=("clinical.example.test",),
                priority=10,
                catalog_version="opaque-query-catalog-v1",
            ),
        ),
        routes=(route,),
        backends=(BackendDefinition("opaque_query_backend", "Opaque Query Backend"),),
    )
    readiness = ReadinessOverlay(
        overlay_version="opaque-query-readiness-v1",
        execution_mode=ExecutionMode.SYNTHETIC,
        routes=(
            RouteReadiness(
                route.route_id,
                ReadinessState.READY,
                CredentialStatus.NOT_REQUIRED,
                "synthetic_ready",
            ),
        ),
    )
    return registry, readiness


def _aggregator_registry(
    *,
    first_predicate: SourcePredicate | None = None,
    second_predicate: SourcePredicate | None = None,
    limits: RouteLimits | None = None,
    max_physical_dispatches: int = 1,
    adapter_id: str = "shared_aggregator_v2",
    adapter_version: str = "synthetic-v1",
) -> tuple[DiscoveryRegistry, ReadinessOverlay]:
    first_predicate = first_predicate or SourcePredicate(
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
        adapter_id=adapter_id,
        route_kind=RouteKind.AGGREGATOR,
        query_modes=(QueryMode.STRUCTURED_QUERY,),
        source_constraint=SourceConstraint.PROVIDER_SOURCE_FILTER,
        attribution_basis="provider_source_field",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=max_physical_dispatches,
        adapter_version=adapter_version,
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


def test_planning_request_accepts_only_exact_public_query_types() -> None:
    queries = (
        "raw structured query",
        planner_module.GeneralFreeTextQuery("Causal inference"),
        planner_module.IdentifierLookupQuery("10.1101/2024.01.02.123456"),
        planner_module.DateIntervalQuery("2024-01-01", "2024-12-31"),
    )

    for query in queries:
        request = _request(("typed_source",), query=query)
        assert request.query is query

    class StringSubclass(str):
        pass

    class GeneralSubclass(planner_module.GeneralFreeTextQuery):
        pass

    for query in (
        StringSubclass("raw"),
        GeneralSubclass("general"),
        SimpleNamespace(text="general"),
        True,
    ):
        with pytest.raises((TypeError, ValueError)):
            _request(("typed_source",), query=query)


@pytest.mark.parametrize(
    "text",
    (
        "",
        "   ",
        "!!!",
        "control\ntext",
        "control\x00text",
        "surrogate\ud800text",
        " ".join(f"term{index}" for index in range(17)),
        "x" * 65,
    ),
)
def test_general_query_rejects_empty_unsafe_or_overlong_term_input(text: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(("typed_source",), query=planner_module.GeneralFreeTextQuery(text))


@pytest.mark.parametrize(
    "text",
    (
        "a" + "!" * 1_039,
        "a" + "\U0001f4a5" * 1_038,
    ),
)
def test_general_query_rejects_raw_character_or_utf8_amplification(text: str) -> None:
    assert len(text) > 1_039 or len(text.encode("utf-8")) > 4_111

    with pytest.raises((TypeError, ValueError)):
        _request(("typed_source",), query=planner_module.GeneralFreeTextQuery(text))


@pytest.mark.parametrize(
    "doi",
    (
        "",
        "10.123/short-registrant",
        "10.1234567890/long-registrant",
        "11.1101/wrong-prefix",
        "10.abc/not-digits",
        "10.1101/no/slash",
        "10.1101/-not-alphanumeric",
        "10.1101/has%percent",
        "10.1101/has\\backslash",
        "10.1101/has space",
        "10.1101/has?query",
        "10.1101/has#fragment",
        "10.1101/café",
        f"10.1101/{'x' * 129}",
    ),
)
def test_identifier_query_rejects_noncanonical_or_path_unsafe_doi(doi: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(("typed_source",), query=planner_module.IdentifierLookupQuery(doi))


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    (
        ("", "2024-01-01"),
        ("2024-1-01", "2024-01-02"),
        ("2024-02-30", "2024-03-01"),
        ("2024-02-02", "2024-02-01"),
        ("2023-01-01", "2024-01-02"),
    ),
)
def test_interval_query_rejects_invalid_reversed_or_overlong_dates(
    start_date: str,
    end_date: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(
            ("typed_source",),
            query=planner_module.DateIntervalQuery(start_date, end_date),
        )


@pytest.mark.parametrize(
    "category",
    (
        "",
        "   ",
        " leading",
        "trailing ",
        "double  space",
        "noncanonical Ａ",
        "punctuation!",
        "---",
        "x" * 129,
    ),
)
def test_category_query_requires_exact_bounded_canonical_category(category: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(
            ("typed_source",),
            query=planner_module.DateIntervalQuery(
                "2024-01-01",
                "2024-01-31",
                category,
            ),
        )


def test_typed_query_rejects_nonempty_filters_and_scalar_subclasses() -> None:
    class StringSubclass(str):
        pass

    typed_queries = (
        planner_module.GeneralFreeTextQuery(StringSubclass("general")),
        planner_module.IdentifierLookupQuery(StringSubclass("10.1101/example")),
        planner_module.DateIntervalQuery(StringSubclass("2024-01-01"), "2024-01-02"),
        planner_module.DateIntervalQuery("2024-01-01", "2024-01-02", StringSubclass("Biology")),
    )
    for query in typed_queries:
        with pytest.raises((TypeError, ValueError)):
            _request(("typed_source",), query=query)

    with pytest.raises((TypeError, ValueError)):
        _request(
            ("typed_source",),
            query=planner_module.GeneralFreeTextQuery("general"),
            filters=(QueryPair("year", "2024"),),
        )


@pytest.mark.parametrize(
    ("query_factory", "selected_route_id", "expected_mode"),
    (
        (
            lambda: planner_module.GeneralFreeTextQuery("Causal inference"),
            "typed_general_search",
            QueryMode.GENERAL_FREE_TEXT,
        ),
        (
            lambda: planner_module.IdentifierLookupQuery("10.1101/2024.01.02.123456"),
            "typed_identifier_lookup",
            QueryMode.IDENTIFIER_LOOKUP,
        ),
        (
            lambda: planner_module.DateIntervalQuery("2024-01-01", "2024-01-31"),
            "typed_interval_browse",
            QueryMode.DATE_INTERVAL,
        ),
        (
            lambda: planner_module.DateIntervalQuery("2024-01-01", "2024-01-31", "Cell Biology"),
            "typed_interval_browse",
            QueryMode.CATEGORY_BROWSE,
        ),
    ),
)
def test_typed_query_selects_only_the_exact_supported_mode(
    query_factory,
    selected_route_id: str,
    expected_mode: QueryMode,
) -> None:
    registry, readiness = _typed_query_registry()
    plan = compile_discovery_plan(
        _request(("typed_source",), query=query_factory(), result_limit=5),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 2, 2, 0, 0, 40_000, 5),
    )

    assert tuple(group.route_id for group in plan.dispatch_groups) == (selected_route_id,)
    assert expected_mode in registry.get_route(selected_route_id).query_modes
    assert tuple(target.route_id for target in plan.skipped) == tuple(
        route.route_id for route in registry.routes if route.route_id != selected_route_id
    )
    assert all(target.status is SkippedStatus.SKIPPED for target in plan.skipped)
    assert all(target.code is SkippedCode.QUERY_MODE_NOT_SUPPORTED for target in plan.skipped)
    assert all(target.reason == "query_mode_not_supported" for target in plan.skipped)


def test_general_query_builds_only_literal_terms_and_route_owned_values() -> None:
    registry, readiness = _typed_query_registry()
    plan = compile_discovery_plan(
        _request(
            ("typed_source",),
            query=planner_module.GeneralFreeTextQuery('  Ｃafé" OR SRC:PPR + beta  '),
            result_limit=75,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 1, 1, 0, 0, 2_000, 50),
    )
    intent = plan.dispatch_groups[0].intents[0]

    assert plan.normalized_query == "Café OR SRC PPR beta"
    assert intent.path == "/europepmc/webservices/rest/search"
    assert intent.query_pairs == (
        QueryPair(
            "query",
            '"Café" AND "OR" AND "SRC" AND "PPR" AND "beta"' ' AND SRC:PPR AND PUBLISHER:"bioRxiv"',
        ),
        QueryPair("format", "json"),
        QueryPair("resultType", "core"),
        QueryPair("pageSize", "50"),
    )
    assert "cursorMark" not in {pair.name for pair in intent.query_pairs}


def test_general_query_omits_named_optional_opaque_cursor_on_first_page() -> None:
    registry, readiness = _opaque_query_registry()
    plan = compile_discovery_plan(
        _request(
            ("opaque_query_source",),
            query=planner_module.GeneralFreeTextQuery("alpha beta"),
            result_limit=100,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 2, 2, 0, 0, 500, 100),
    )

    assert plan.dispatch_groups[0].intents[0].query_pairs == (
        QueryPair("query.term", '"alpha" AND "beta"'),
        QueryPair("format", "json"),
        QueryPair("pageSize", "50"),
    )


@pytest.mark.parametrize(
    ("route_max_results", "expected_page_size"),
    ((100, "50"), (25, "25")),
)
def test_general_query_clamps_decimal_to_route_and_rule_ceilings(
    route_max_results: int,
    expected_page_size: str,
) -> None:
    registry, readiness = _opaque_query_registry()
    route = registry.get_route("opaque_query_search")
    registry = replace(
        registry,
        routes=(
            replace(
                route,
                policy=replace(
                    route.policy,
                    limits=replace(route.policy.limits, max_results=route_max_results),
                    policy_digest="",
                ),
            ),
        ),
    )
    plan = compile_discovery_plan(
        _request(
            ("opaque_query_source",),
            query=planner_module.GeneralFreeTextQuery("alpha beta"),
            result_limit=100,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 2, 2, 0, 0, 500, 100),
    )

    assert plan.dispatch_groups[0].intents[0].query_pairs[-1] == QueryPair("pageSize", expected_page_size)


@pytest.mark.parametrize(
    "query_value_policies",
    (
        (
            LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
            OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=True),
            ExactQueryValuePolicy("format", "json"),
            BoundedDecimalQueryValuePolicy("pageSize", 50),
        ),
        (
            LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
            ExactQueryValuePolicy("pageToken", "first"),
            OpaqueCursorQueryValuePolicy("format", 1_024, required=False),
            BoundedDecimalQueryValuePolicy("pageSize", 50),
        ),
        (
            LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
            OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=False),
            OpaqueCursorQueryValuePolicy("format", 1_024, required=False),
            BoundedDecimalQueryValuePolicy("pageSize", 50),
        ),
    ),
)
def test_general_query_rejects_invalid_optional_opaque_cursor_policy(
    query_value_policies: tuple[object, ...],
) -> None:
    registry, readiness = _opaque_query_registry()
    route = registry.get_route("opaque_query_search")
    registry = replace(
        registry,
        routes=(
            replace(
                route,
                policy=replace(route.policy, query_value_policies=query_value_policies, policy_digest=""),
            ),
        ),
    )

    with pytest.raises(PlanningError) as exc_info:
        compile_discovery_plan(
            _request(
                ("opaque_query_source",),
                query=planner_module.GeneralFreeTextQuery("alpha beta"),
                result_limit=100,
            ),
            registry=registry,
            readiness=readiness,
            budget=BudgetCeilings(1, 2, 2, 0, 0, 500, 100),
        )

    assert exc_info.value.code == "invalid_optional_opaque_cursor_policy:opaque_query_search"


def test_general_query_rejects_term_longer_than_route_literal_policy_before_emitting_intent() -> None:
    registry, readiness = _opaque_query_registry()

    with pytest.raises(PlanningError) as exc_info:
        compile_discovery_plan(
            _request(
                ("opaque_query_source",),
                query=planner_module.GeneralFreeTextQuery("a" * 33),
                result_limit=100,
            ),
            registry=registry,
            readiness=readiness,
            budget=BudgetCeilings(1, 2, 2, 0, 0, 500, 100),
        )

    assert exc_info.value.code == "invalid_literal_terms_policy:opaque_query_search"


def test_general_query_accepts_exact_sixteen_by_sixty_four_term_boundary() -> None:
    registry, readiness = _typed_query_registry()
    text = " ".join(chr(ord("a") + index) * 64 for index in range(16))
    plan = compile_discovery_plan(
        _request(
            ("typed_source",),
            query=planner_module.GeneralFreeTextQuery(text),
            result_limit=1,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 1, 1, 0, 0, 2_000, 1),
    )

    expression = plan.dispatch_groups[0].intents[0].query_pairs[0].value
    assert plan.normalized_query == text
    assert len(plan.normalized_query) == 16 * 64 + 15 == 1_039
    assert len(plan.normalized_query.encode("utf-8")) <= 16 * 64 * 4 + 15 == 4_111
    assert expression.count('" AND "') == 15
    assert expression.endswith(' AND SRC:PPR AND PUBLISHER:"bioRxiv"')


def test_general_query_persists_only_bounded_alphanumeric_terms() -> None:
    registry, readiness = _typed_query_registry()
    plan = compile_discovery_plan(
        _request(
            ("typed_source",),
            query=planner_module.GeneralFreeTextQuery("alpha" + "!" * 1_000 + "beta"),
            result_limit=1,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 1, 1, 0, 0, 2_000, 1),
    )

    assert plan.normalized_query == "alpha beta"
    assert len(plan.normalized_query) <= 1_039
    assert len(plan.normalized_query.encode("utf-8")) <= 4_111
    assert plan.dispatch_groups[0].intents[0].query_pairs[0].value == (
        '"alpha" AND "beta" AND SRC:PPR AND PUBLISHER:"bioRxiv"'
    )


def test_identifier_query_renders_two_canonical_dynamic_path_segments() -> None:
    registry, readiness = _typed_query_registry()
    plan = compile_discovery_plan(
        _request(
            ("typed_source",),
            query=planner_module.IdentifierLookupQuery("10.1101/abc(def):ghi"),
            result_limit=5,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 1, 1, 0, 0, 2_000, 5),
    )
    intent = plan.dispatch_groups[0].intents[0]

    assert plan.normalized_query == "10.1101/abc(def):ghi"
    assert intent.path == "/details/biorxiv/10.1101/abc%28def%29%3Aghi/na/json"
    assert intent.query_pairs == ()


def test_identifier_query_lowercases_doi_plan_path_and_group_identity() -> None:
    registry, readiness = _typed_query_registry()

    def compile_doi(doi: str):
        return compile_discovery_plan(
            _request(
                ("typed_source",),
                query=planner_module.IdentifierLookupQuery(doi),
                result_limit=5,
            ),
            registry=registry,
            readiness=readiness,
            budget=BudgetCeilings(1, 1, 1, 0, 0, 2_000, 5),
        )

    mixed_case_plan = compile_doi("10.1101/ABC.Def")
    lowercase_plan = compile_doi("10.1101/abc.def")
    mixed_case_group = mixed_case_plan.dispatch_groups[0]
    lowercase_group = lowercase_plan.dispatch_groups[0]

    assert mixed_case_plan.normalized_query == lowercase_plan.normalized_query == "10.1101/abc.def"
    assert mixed_case_group.intents[0].path == "/details/biorxiv/10.1101/abc.def/na/json"
    assert mixed_case_group.intents == lowercase_group.intents
    assert mixed_case_group.dispatch_group_id == lowercase_group.dispatch_group_id


@pytest.mark.parametrize(
    ("category", "expected_pairs"),
    (
        (None, ()),
        ("Cell Biology & Genomics/Genetics", (QueryPair("category", "Cell Biology & Genomics/Genetics"),)),
    ),
)
def test_interval_query_renders_fixed_server_dates_and_initial_path_cursor(
    category: str | None,
    expected_pairs: tuple[QueryPair, ...],
) -> None:
    registry, readiness = _typed_query_registry()
    plan = compile_discovery_plan(
        _request(
            ("typed_source",),
            query=planner_module.DateIntervalQuery("2024-01-01", "2024-12-31", category),
            result_limit=5,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 2, 2, 0, 0, 4_000, 5),
    )
    intent = plan.dispatch_groups[0].intents[0]

    assert intent.path == "/details/biorxiv/2024-01-01/2024-12-31/0/json"
    assert intent.query_pairs == expected_pairs


@pytest.mark.parametrize(
    "query",
    (
        planner_module.DateIntervalQuery("2024-02-29", "2024-02-29"),
        planner_module.DateIntervalQuery("2024-01-01", "2024-12-31"),
        planner_module.DateIntervalQuery("2023-01-01", "2024-01-01"),
        planner_module.DateIntervalQuery("2024-01-01", "2024-01-01", "A" * 128),
    ),
)
def test_interval_and_category_accept_exact_calendar_and_length_boundaries(query: object) -> None:
    request = _request(("typed_source",), query=query)
    assert request.query is query


def test_raw_string_selects_structured_mode_and_preserves_foundation_behavior() -> None:
    registry, readiness = _typed_query_registry()
    typed_family_plan = compile_discovery_plan(
        _request(("typed_source",), query="raw structured query", result_limit=5),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(0, 0, 0, 0, 0, 0, 0),
    )

    assert typed_family_plan.dispatch_groups == ()
    assert tuple(target.route_id for target in typed_family_plan.skipped) == tuple(
        route.route_id for route in registry.routes
    )
    assert all(target.code is SkippedCode.QUERY_MODE_NOT_SUPPORTED for target in typed_family_plan.skipped)

    foundation_plan = compile_discovery_plan(
        _request(("arxiv",), query="  Causal   Inference  "),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_budget(),
    )
    assert foundation_plan.normalized_query == "causal inference"
    assert foundation_plan.dispatch_groups[0].intents[0].query_pairs[0] == QueryPair(
        "search_query",
        "all:causal inference",
    )


def test_raw_structured_query_with_lone_surrogate_raises_stable_planning_error() -> None:
    with pytest.raises(PlanningError) as exc_info:
        compile_discovery_plan(
            _request(("arxiv",), query="\ud800"),
            registry=foundation_registry(),
            readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
            budget=_budget(),
        )

    assert exc_info.value.code == "query_contains_invalid_unicode"


@pytest.mark.parametrize("incompatible_state", ("missing", "credentialed", "disabled"))
def test_mode_mismatch_precedes_credential_and_readiness_checks(incompatible_state: str) -> None:
    registry, readiness = _typed_query_registry()
    general_route = registry.get_route("typed_general_search")
    if incompatible_state == "credentialed":
        registry = replace(
            registry,
            routes=tuple(
                (
                    replace(route, credential_requirement=CredentialRequirement.API_KEY)
                    if route.route_id == general_route.route_id
                    else route
                )
                for route in registry.routes
            ),
        )
    elif incompatible_state == "disabled":
        readiness = replace(
            readiness,
            routes=tuple(
                (
                    replace(entry, state=ReadinessState.DISABLED, reason="must_not_win")
                    if entry.route_id == general_route.route_id
                    else entry
                )
                for entry in readiness.routes
            ),
        )
    else:
        readiness = replace(
            readiness,
            routes=tuple(entry for entry in readiness.routes if entry.route_id != general_route.route_id),
        )

    plan = compile_discovery_plan(
        _request(
            ("typed_source",),
            query=planner_module.IdentifierLookupQuery("10.1101/example"),
            result_limit=5,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 1, 1, 0, 0, 20_000, 5),
    )

    skipped = next(target for target in plan.skipped if target.route_id == general_route.route_id)
    assert skipped.status is SkippedStatus.SKIPPED
    assert skipped.code is SkippedCode.QUERY_MODE_NOT_SUPPORTED
    assert skipped.reason == "query_mode_not_supported"


def test_openalex_is_typed_unavailable_with_no_attempt_or_dispatch_allowance() -> None:
    plan = compile_discovery_plan(
        _request(("openalex",)),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_budget(),
    )

    assert plan.dispatch_groups == ()
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

    assert plan.dispatch_groups == ()
    assert plan.allowance.physical_dispatches == 0
    assert plan.skipped[0].code is SkippedCode.CREDENTIALED_OUT_OF_SCOPE


def test_foundation_plan_compiles_seven_routes_and_pubmed_two_dispatch_allowance() -> None:
    plan = compile_discovery_plan(
        _request(_FOUNDATION_SOURCE_IDS),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.OFFLINE_FIXTURE),
        budget=_budget(),
    )

    assert [group.backend_id for group in plan.dispatch_groups] == [
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


@pytest.mark.parametrize(
    ("source_ids", "result_limit", "capacity", "expected"),
    [
        (("arxiv",), 101, 100, 100),
        (("arxiv", "crossref"), 150, 200, 150),
        (("arxiv", "crossref"), 250, 200, 200),
        (("pubmed",), 101, 100, 100),
        (
            ("semantic_scholar", "crossref", "arxiv", "pubmed", "zenodo", "figshare", "osf"),
            25,
            625,
            25,
        ),
    ],
)
def test_returned_results_are_global_post_executor_cap_not_raw_candidate_capacity(
    source_ids: tuple[str, ...],
    result_limit: int,
    capacity: int,
    expected: int,
) -> None:
    registry = foundation_registry()
    plan = compile_discovery_plan(
        _request(source_ids, result_limit=result_limit),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(max_results=expected),
    )

    raw_candidate_capacity = sum(group.limits.max_results for group in plan.dispatch_groups)
    assert raw_candidate_capacity == capacity
    assert plan.allowance.returned_results == expected


def test_pubmed_declares_two_possible_dispatches_without_runtime_accounting() -> None:
    plan = compile_discovery_plan(
        _request(("pubmed",)),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    group = plan.dispatch_groups[0]

    assert group.allowance.physical_dispatches == 2
    assert [intent.operation_kind.value for intent in group.intents] == [
        "search",
        "conditional_summary",
    ]
    assert [intent.path for intent in group.intents] == [
        "/entrez/eutils/esearch.fcgi",
        "/entrez/eutils/esummary.fcgi",
    ]
    summary = group.intents[1]
    assert summary.query_pairs == (QueryPair("db", "pubmed"), QueryPair("retmode", "json"))
    assert summary.query_bindings == (
        DeferredNumericCSVQueryBinding(
            binding_id="pubmed_esearch_ids",
            query_name="id",
            max_items=25,
            max_item_chars=16,
        ),
    )
    assert "{esearch_ids}" not in canonical_plan_bytes(plan).decode("utf-8")
    assert not any(hasattr(group.allowance, name) for name in ("reservation", "reserved", "debit", "release"))


def test_figshare_plan_uses_official_query_and_json_body_shape() -> None:
    plan = compile_discovery_plan(
        _request(("figshare",), query="Causal Inference", result_limit=25),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    intent = plan.dispatch_groups[0].intents[0]

    assert intent.method == "POST"
    assert intent.query_pairs == ()
    assert intent.json_body_pairs == (
        JSONBodyPair("search_for", "causal inference"),
        JSONBodyPair("page", 1),
        JSONBodyPair("page_size", 25),
    )
    assert intent.query_bindings == ()


@pytest.mark.parametrize("requested_limit", (25, 26, 100))
def test_zenodo_anonymous_search_clamps_physical_page_size_to_twenty_five(requested_limit: int) -> None:
    plan = compile_discovery_plan(
        _request(("zenodo",), query="Causal Inference", result_limit=requested_limit),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    group = plan.dispatch_groups[0]

    assert plan.result_limit == requested_limit
    assert group.limits.max_results == 25
    assert group.intents[0].query_pairs == (
        QueryPair("q", "causal inference"),
        QueryPair("page", "1"),
        QueryPair("size", "25"),
    )
    assert plan.allowance.returned_results == 25


def test_semantic_scholar_plan_requests_exact_consumed_fields() -> None:
    plan = compile_discovery_plan(
        _request(("semantic_scholar",), query="Causal Inference", result_limit=25),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )

    assert plan.dispatch_groups[0].intents[0].query_pairs == (
        QueryPair("query", "causal inference"),
        QueryPair("offset", "0"),
        QueryPair("limit", "25"),
        QueryPair(
            "fields",
            "paperId,title,authors,abstract,tldr,externalIds,url,openAccessPdf",
        ),
    )


def test_crossref_plan_requests_exact_consumed_fields() -> None:
    plan = compile_discovery_plan(
        _request(("crossref",), query="Causal Inference", result_limit=25),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )

    assert plan.dispatch_groups[0].intents[0].query_pairs == (
        QueryPair("query", "causal inference"),
        QueryPair("offset", "0"),
        QueryPair("rows", "25"),
        QueryPair("select", "DOI,title,author,abstract,URL,link"),
    )


def test_osf_plan_uses_exact_title_filter_and_plain_page_number_shape() -> None:
    plan = compile_discovery_plan(
        _request(("osf",), query="Causal Inference", result_limit=25),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )

    assert plan.dispatch_groups[0].intents[0].query_pairs == (
        QueryPair("filter[title]", "causal inference"),
        QueryPair("page", "1"),
        QueryPair("page[size]", "25"),
    )
    assert {pair.name for pair in plan.dispatch_groups[0].intents[0].query_pairs}.isdisjoint({"q", "filter"})


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
    assert tuple(group.dispatch_group_id for group in first.dispatch_groups) == tuple(
        group.dispatch_group_id for group in second.dispatch_groups
    )


def test_compiled_plan_digest_is_deterministic_and_excludes_live_ceilings() -> None:
    registry = foundation_registry()
    readiness = foundation_readiness(ExecutionMode.SYNTHETIC)
    first = compile_discovery_plan(
        _request(("crossref", "semantic_scholar"), query="  Causal   Inference  "),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    second = compile_discovery_plan(
        _request(("semantic_scholar", "crossref"), query="causal inference"),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    first_digest = first.plan_digest

    assert isinstance(first_digest, str)
    assert len(first_digest) == 64
    assert first_digest == second.plan_digest
    assert canonical_plan_digest(first) == first_digest
    object.__setattr__(first.ceilings, "max_results", 1)
    assert canonical_plan_digest(first) == first_digest


def test_compiled_ids_equal_shared_typed_recomputation() -> None:
    plan = compile_discovery_plan(
        _request(_FOUNDATION_SOURCE_IDS),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    for group in plan.dispatch_groups:
        recomputed_group_id = expected_dispatch_group_id(group)
        assert group.dispatch_group_id == recomputed_group_id
        assert all(
            attempt.logical_attempt_id == expected_logical_attempt_id(attempt, recomputed_group_id)
            for attempt in group.logical_attempts
        )


def test_shared_id_recomputation_changes_with_hashed_group_and_logical_material() -> None:
    group = compile_discovery_plan(
        _request(("figshare",), filters=(QueryPair("year", "2025"),)),
        registry=foundation_registry(),
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    ).dispatch_groups[0]
    baseline_group_id = expected_dispatch_group_id(group)
    intent = group.intents[0]
    query_changed = replace(
        group,
        intents=(
            replace(
                intent,
                query_pairs=(QueryPair("page", "24"),),
            ),
        ),
    )
    body_changed = replace(
        group,
        intents=(
            replace(
                intent,
                json_body_pairs=(
                    replace(intent.json_body_pairs[0], value="changed query"),
                    *intent.json_body_pairs[1:],
                ),
            ),
        ),
    )
    filters_changed = replace(group, filters=(QueryPair("year", "2024"),))

    assert expected_dispatch_group_id(query_changed) != baseline_group_id
    assert expected_dispatch_group_id(body_changed) != baseline_group_id
    assert expected_dispatch_group_id(filters_changed) != baseline_group_id

    attempt = group.logical_attempts[0]
    selection_changed = replace(attempt, selection_reason="explicitly changed selection")
    assert expected_logical_attempt_id(selection_changed, baseline_group_id) != attempt.logical_attempt_id


def test_canonical_predicate_values_produce_equal_plan_bytes_and_attempt_ids() -> None:
    first_registry, first_readiness = _aggregator_registry(
        first_predicate=SourcePredicate(
            field_path=("source", "collection"),
            operator=PredicateOperator.EQUALS_ANY,
            values=(" Shared   Index ",),
        )
    )
    second_registry, second_readiness = _aggregator_registry(
        first_predicate=SourcePredicate(
            field_path=("source", "collection"),
            operator=PredicateOperator.EQUALS_ANY,
            values=("shared index",),
        )
    )

    first = compile_discovery_plan(
        _request(("target_a",)),
        registry=first_registry,
        readiness=first_readiness,
        budget=_budget(),
    )
    second = compile_discovery_plan(
        _request(("target_a",)),
        registry=second_registry,
        readiness=second_readiness,
        budget=_budget(),
    )

    assert canonical_plan_bytes(first) == canonical_plan_bytes(second)
    assert first.dispatch_groups[0].dispatch_group_id == second.dispatch_groups[0].dispatch_group_id
    assert (
        first.dispatch_groups[0].logical_attempts[0].logical_attempt_id
        == second.dispatch_groups[0].logical_attempts[0].logical_attempt_id
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
    assert len(plan.dispatch_groups) == 1
    assert [attempt.catalog_source_id for attempt in plan.dispatch_groups[0].logical_attempts] == [
        "target_a",
        "target_b",
    ]
    assert plan.dispatch_groups[0].backend_id == "shared_aggregator"
    assert [attempt.source_predicate for attempt in plan.dispatch_groups[0].logical_attempts] == [
        registry.get_source("target_a").route_references[0].source_predicate,
        second_predicate,
    ]


def test_compiled_predicates_are_isolated_from_registry_and_each_other() -> None:
    registry, readiness = _aggregator_registry()
    first_registry_predicate = registry.get_source("target_a").route_references[0].source_predicate
    second_registry_predicate = registry.get_source("target_b").route_references[0].source_predicate
    assert first_registry_predicate is not None
    assert first_registry_predicate is second_registry_predicate

    plan = compile_discovery_plan(
        _request(("target_a", "target_b")),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    first_planned_predicate = plan.dispatch_groups[0].logical_attempts[0].source_predicate
    second_planned_predicate = plan.dispatch_groups[0].logical_attempts[1].source_predicate

    assert first_planned_predicate == first_registry_predicate
    assert second_planned_predicate == first_registry_predicate
    assert first_planned_predicate is not first_registry_predicate
    assert second_planned_predicate is not first_registry_predicate
    assert first_planned_predicate is not second_planned_predicate

    object.__setattr__(first_planned_predicate, "values", ("attacker",))

    assert first_registry_predicate.values == ("shared-index",)
    assert second_planned_predicate.values == ("shared-index",)


def test_coalescing_preserves_physical_and_per_target_logical_identity() -> None:
    registry, readiness = _aggregator_registry()
    single = compile_discovery_plan(
        _request(("target_a",)),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    coalesced = compile_discovery_plan(
        _request(("target_a", "target_b")),
        registry=registry,
        readiness=readiness,
        budget=_budget(),
    )
    single_group = single.dispatch_groups[0]
    coalesced_group = coalesced.dispatch_groups[0]
    coalesced_ids = {
        attempt.catalog_source_id: attempt.logical_attempt_id for attempt in coalesced_group.logical_attempts
    }

    assert single_group.dispatch_group_id == coalesced_group.dispatch_group_id
    assert single_group.logical_attempts[0].logical_attempt_id == coalesced_ids["target_a"]
    assert len(set(coalesced_ids.values())) == 2
    assert single.allowance.route_attempts == 1
    assert coalesced.allowance.route_attempts == 2
    assert single.allowance.physical_dispatches == coalesced.allowance.physical_dispatches == 1


def test_dispatch_group_freezes_adapter_identity_and_hashes_adapter_revisions() -> None:
    baseline_registry, baseline_readiness = _aggregator_registry()
    revised_id_registry, revised_id_readiness = _aggregator_registry(adapter_id="shared_aggregator_v3")
    revised_version_registry, revised_version_readiness = _aggregator_registry(adapter_version="synthetic-v2")

    def compile_group(registry: DiscoveryRegistry, readiness: ReadinessOverlay) -> PlannedDispatchGroup:
        return compile_discovery_plan(
            _request(("target_a",)),
            registry=registry,
            readiness=readiness,
            budget=_budget(),
        ).dispatch_groups[0]

    baseline = compile_group(baseline_registry, baseline_readiness)
    revised_id = compile_group(revised_id_registry, revised_id_readiness)
    revised_version = compile_group(revised_version_registry, revised_version_readiness)
    route = baseline_registry.routes[0]

    assert baseline.adapter_id == route.adapter_id
    assert baseline.adapter_version == route.adapter_version
    assert baseline.policy_digest == revised_id.policy_digest == revised_version.policy_digest
    assert baseline.dispatch_group_id != revised_id.dispatch_group_id
    assert baseline.dispatch_group_id != revised_version.dispatch_group_id


def test_coalesced_targets_count_one_physical_route_result_capacity() -> None:
    registry, readiness = _aggregator_registry()
    plan = compile_discovery_plan(
        _request(("target_a", "target_b"), result_limit=75),
        registry=registry,
        readiness=readiness,
        budget=_budget(max_results=50),
    )

    assert len(plan.dispatch_groups) == 1
    assert plan.allowance.returned_results == 50


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
    targets = plan.dispatch_groups[0].logical_attempts

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

    assert plan.dispatch_groups[0].allowance.physical_dispatches == 4
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

    assert [(group.route_id, group.fallback_order) for group in plan.dispatch_groups] == [
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
