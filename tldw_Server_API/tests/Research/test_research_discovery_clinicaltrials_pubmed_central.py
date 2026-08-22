"""Task 5 contracts for the shadow NCBI identity overlay."""

from __future__ import annotations

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
    BudgetCeilings,
    ExecutionMode,
    QueryPair,
)
from tldw_Server_API.app.core.Research.discovery.planner import (
    PlanningRequest,
    compile_discovery_plan,
)
from tldw_Server_API.app.core.Research.discovery.registry import foundation_readiness

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
