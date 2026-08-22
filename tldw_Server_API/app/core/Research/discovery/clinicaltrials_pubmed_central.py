"""Shadow-only ClinicalTrials.gov and PubMed Central discovery family."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from .contracts import AccessRoute
from .gateway_adapters import _ParsingProfile
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


def clinicaltrials_pubmed_central_shadow_registry() -> DiscoveryRegistry:
    """Return the foundation registry with its PubMed identity overlay only."""
    foundation = foundation_registry()
    routes = tuple(
        _pubmed_identity_overlay(route) if route.route_id == _PUBMED_ROUTE_ID else route for route in foundation.routes
    )
    return DiscoveryRegistry(
        catalog_version=SHADOW_CATALOG_VERSION,
        registry_version=SHADOW_REGISTRY_VERSION,
        sources=tuple(replace(source, catalog_version=SHADOW_CATALOG_VERSION) for source in foundation.sources),
        routes=routes,
        backends=foundation.backends,
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
