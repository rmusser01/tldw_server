"""Task 5 contracts for the shadow NCBI identity overlay."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Research.discovery.clinicaltrials_pubmed_central import (
    _FAMILY_PARSING_PROFILES,
    CLINICALTRIALS_FIELDS,
    CLINICALTRIALS_GOV_ADAPTER_ID,
    CLINICALTRIALS_GOV_ADAPTER_VERSION,
    NCBI_EMAIL,
    NCBI_TOOL,
    PUBMED_CENTRAL_ADAPTER_ID,
    PUBMED_CENTRAL_ADAPTER_VERSION,
    PUBMED_IDENTITY_ADAPTER_VERSION,
    PUBMED_IDENTITY_POLICY_VERSION,
    ROUTE_POLICY_VERSION,
    SHADOW_CATALOG_VERSION,
    SHADOW_READINESS_VERSION,
    SHADOW_REGISTRY_VERSION,
    clinicaltrials_pubmed_central_shadow_registry,
)
from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    BudgetCeilings,
    ExecutionMode,
    QueryPair,
)
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningError,
    PlanningRequest,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)

pytestmark = pytest.mark.unit


def _budget() -> BudgetCeilings:
    return BudgetCeilings(1, 2, 1, 0, 0, 40_000, 10)


def test_shadow_registry_replaces_only_pubmed_with_the_identity_overlay() -> None:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    route = registry.get_route("pubmed_ncbi_eutils_pubmed_direct")

    assert (
        SHADOW_CATALOG_VERSION,
        SHADOW_REGISTRY_VERSION,
        SHADOW_READINESS_VERSION,
        ROUTE_POLICY_VERSION,
        CLINICALTRIALS_GOV_ADAPTER_ID,
        CLINICALTRIALS_GOV_ADAPTER_VERSION,
        PUBMED_CENTRAL_ADAPTER_ID,
        PUBMED_CENTRAL_ADAPTER_VERSION,
        PUBMED_IDENTITY_POLICY_VERSION,
        PUBMED_IDENTITY_ADAPTER_VERSION,
        NCBI_TOOL,
        NCBI_EMAIL,
    ) == (
        "research-discovery-v2-clinicaltrials-pmc-shadow",
        "research-discovery-v2-clinicaltrials-pmc-shadow-2026-08-21",
        "research-discovery-readiness-v2-clinicaltrials-pmc-shadow",
        "research-discovery-route-policy-v2-clinicaltrials-pmc",
        "clinicaltrials_gov_v2",
        "clinicaltrials-gov-v2",
        "pubmed_central_v2",
        "pubmed-central-v2",
        "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21",
        "pubmed-v2-ncbi-identity",
        "tldw_server",
        "contact@tldwproject.com",
    )
    assert CLINICALTRIALS_FIELDS == (
        "NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,"
        "InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults"
    )
    assert registry.catalog_version == SHADOW_CATALOG_VERSION
    assert all(source.catalog_version == SHADOW_CATALOG_VERSION for source in registry.sources)
    assert route.adapter_version == PUBMED_IDENTITY_ADAPTER_VERSION
    assert route.policy.policy_version == PUBMED_IDENTITY_POLICY_VERSION
    assert route.policy.allowed_query_keys == (
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
    assert route.policy.pagination_query_key == "retstart"
    assert route.policy.query_value_policies == ()
    assert set(_FAMILY_PARSING_PROFILES) == {
        (CLINICALTRIALS_GOV_ADAPTER_ID, CLINICALTRIALS_GOV_ADAPTER_VERSION),
        (PUBMED_CENTRAL_ADAPTER_ID, PUBMED_CENTRAL_ADAPTER_VERSION),
    }
    assert {
        identity: (
            profile.max_input_bytes,
            profile.max_records,
            profile.max_depth,
            profile.max_nodes,
            profile.max_string_chars,
            profile.max_numeric_token_chars,
            profile.parse_deadline_ms,
        )
        for identity, profile in _FAMILY_PARSING_PROFILES.items()
    } == {
        ("clinicaltrials_gov_v2", "clinicaltrials-gov-v2"): (2_097_152, 50, 16, 50_000, 65_536, 32, 500),
        ("pubmed_central_v2", "pubmed-central-v2"): (2_097_152, 100, 16, 50_000, 65_536, 32, 500),
    }


def test_shadow_registry_preserves_every_non_pubmed_route_exactly() -> None:
    foundation = foundation_registry()
    shadow = clinicaltrials_pubmed_central_shadow_registry()

    assert tuple(route for route in shadow.routes if route.route_id != "pubmed_ncbi_eutils_pubmed_direct") == tuple(
        route for route in foundation.routes if route.route_id != "pubmed_ncbi_eutils_pubmed_direct"
    )


def test_identity_overlay_plans_exact_identity_pairs_on_both_hops() -> None:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    plan = compile_discovery_plan(
        PlanningRequest(("pubmed",), "bounded discovery", (), 7),
        registry=registry,
        readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
        budget=_budget(),
    )
    search, summary = plan.dispatch_groups[0].intents

    assert tuple((pair.name, pair.value) for pair in search.query_pairs) == (
        ("db", "pubmed"),
        ("term", "bounded discovery"),
        ("retstart", "0"),
        ("retmax", "7"),
        ("retmode", "json"),
        ("sort", "relevance"),
        ("tool", "tldw_server"),
        ("email", "contact@tldwproject.com"),
    )
    assert tuple((pair.name, pair.value) for pair in summary.query_pairs) == (
        ("db", "pubmed"),
        ("retmode", "json"),
        ("tool", "tldw_server"),
        ("email", "contact@tldwproject.com"),
    )
    assert summary.query_bindings[0].query_name == "id"
    assert QueryPair("tool", "tldw_server") in search.query_pairs


def _registry_with_pubmed_route(route: AccessRoute) -> DiscoveryRegistry:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    original_route_id = "pubmed_ncbi_eutils_pubmed_direct"
    return DiscoveryRegistry(
        catalog_version=registry.catalog_version,
        registry_version="identity-mutation-registry-v1",
        sources=tuple(
            replace(
                source,
                route_references=tuple(
                    (
                        replace(reference, route_id=route.route_id)
                        if reference.route_id == original_route_id
                        else reference
                    )
                    for reference in source.route_references
                ),
            )
            for source in registry.sources
        ),
        routes=tuple(route if item.route_id == original_route_id else item for item in registry.routes),
        backends=registry.backends,
    )


def _generic_registry_with_identity_component(component: str) -> tuple[DiscoveryRegistry, AccessRoute]:
    foundation = foundation_registry()
    source = foundation.get_source("arxiv")
    original = foundation.get_route(source.route_references[0].route_id)
    if component == "adapter_version":
        mutated = replace(original, adapter_version=PUBMED_IDENTITY_ADAPTER_VERSION)
    elif component == "policy_version":
        mutated = replace(
            original,
            policy=replace(
                original.policy,
                policy_version=PUBMED_IDENTITY_POLICY_VERSION,
                policy_digest="",
            ),
        )
    else:
        raise ValueError("unknown_identity_component")
    registry = DiscoveryRegistry(
        catalog_version=foundation.catalog_version,
        registry_version="generic-identity-component-registry-v1",
        sources=foundation.sources,
        routes=tuple(mutated if route.route_id == original.route_id else route for route in foundation.routes),
        backends=foundation.backends,
    )
    return registry, mutated


def test_identity_adapter_version_on_generic_route_fails_closed_before_plan_emission() -> None:
    registry, route = _generic_registry_with_identity_component("adapter_version")

    assert route.route_id != "pubmed_ncbi_eutils_pubmed_direct"
    assert route.backend_id != "ncbi_eutils_pubmed"
    assert route.adapter_id != "pubmed_v2"
    with pytest.raises(PlanningError, match="invalid_pubmed_route_identity"):
        compile_discovery_plan(
            PlanningRequest(("arxiv",), "bounded discovery", (), 7),
            registry=registry,
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=_budget(),
        )


def test_identity_policy_version_on_generic_route_fails_closed_before_plan_emission() -> None:
    registry, route = _generic_registry_with_identity_component("policy_version")

    assert route.route_id != "pubmed_ncbi_eutils_pubmed_direct"
    assert route.backend_id != "ncbi_eutils_pubmed"
    assert route.adapter_id != "pubmed_v2"
    with pytest.raises(PlanningError, match="invalid_pubmed_route_identity"):
        compile_discovery_plan(
            PlanningRequest(("arxiv",), "bounded discovery", (), 7),
            registry=registry,
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=_budget(),
        )


@pytest.mark.parametrize(
    "route_change",
    (
        {"route_id": "pubmed_ncbi_eutils_pubmed_partial"},
        {"backend_id": "crossref_api"},
        {"adapter_id": "pubmed_v3"},
        {"adapter_version": "foundation-v2"},
        {"policy": "foundation"},
    ),
)
def test_partial_or_swapped_pubmed_overlay_identity_fails_closed_before_plan_emission(
    route_change: dict[str, str],
) -> None:
    registry = clinicaltrials_pubmed_central_shadow_registry()
    route = registry.get_route("pubmed_ncbi_eutils_pubmed_direct")
    policy = route.policy
    if route_change.get("policy") == "foundation":
        policy = replace(policy, policy_version="research-discovery-route-policy-v2-foundation", policy_digest="")
    mutated = replace(
        route,
        route_id=route_change.get("route_id", route.route_id),
        backend_id=route_change.get("backend_id", route.backend_id),
        adapter_id=route_change.get("adapter_id", route.adapter_id),
        adapter_version=route_change.get("adapter_version", route.adapter_version),
        policy=policy,
    )
    readiness = foundation_readiness(ExecutionMode.SYNTHETIC)
    readiness = replace(
        readiness,
        routes=tuple(
            replace(entry, route_id=mutated.route_id) if entry.route_id == route.route_id else entry
            for entry in readiness.routes
        ),
    )

    with pytest.raises(PlanningError, match="invalid_pubmed_route_identity"):
        compile_discovery_plan(
            PlanningRequest(("pubmed",), "bounded discovery", (), 7),
            registry=_registry_with_pubmed_route(mutated),
            readiness=readiness,
            budget=_budget(),
        )


@pytest.mark.parametrize(
    "filters",
    (
        (QueryPair("tool", "attacker"),),
        (QueryPair("email", "attacker@example.test"),),
        (QueryPair("tool", "attacker"), QueryPair("email", "attacker@example.test")),
    ),
)
def test_identity_overlay_rejects_user_supplied_identity_filters(filters: tuple[QueryPair, ...]) -> None:
    with pytest.raises(PlanningError, match="identity_query_filter_not_allowed"):
        compile_discovery_plan(
            PlanningRequest(("pubmed",), "bounded discovery", filters, 7),
            registry=clinicaltrials_pubmed_central_shadow_registry(),
            readiness=foundation_readiness(ExecutionMode.SYNTHETIC),
            budget=_budget(),
        )
