"""Reconcile the V2 foundation registry with the frozen source inventory."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
from tldw_Server_API.app.core.Research.discovery.contracts import (
    BackendDefinition,
    CredentialRequirement,
    CredentialStatus,
    ExecutionMode,
    QueryMode,
    ReadinessState,
    RouteKind,
    SourceConstraint,
    SourceDefinition,
    SourceRouteReference,
)
from tldw_Server_API.app.core.Research.discovery.registry import (
    DiscoveryRegistry,
    foundation_readiness,
    foundation_registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEDGER_PATH = (
    _REPO_ROOT / "Docs" / "Design" / "research_source_inventory" / "research-source-coverage-ledger-2026-07-13.json"
)

_EXPECTED_LEDGER_ROUTES = {
    "arxiv": (
        "sourclip-2026-07-13-0002",
        "arxiv_arxiv_api_direct",
        "arxiv_api",
        "none",
    ),
    "pubmed": (
        "sourclip-2026-07-13-0003",
        "pubmed_ncbi_eutils_pubmed_direct",
        "ncbi_eutils_pubmed",
        "none",
    ),
    "semantic_scholar": (
        "sourclip-2026-07-13-0006",
        "semantic_scholar_semantic_scholar_graph_api_direct",
        "semantic_scholar_graph_api",
        "none",
    ),
    "zenodo": (
        "sourclip-2026-07-13-0071",
        "zenodo_zenodo_records_api_direct",
        "zenodo_records_api",
        "none",
    ),
    "openalex": (
        "sourclip-2026-07-13-0088",
        "openalex_openalex_api_direct",
        "openalex_api",
        "api_key",
    ),
    "osf": (
        "sourclip-2026-07-13-0132",
        "open_science_framework_osf_api_direct",
        "osf_api",
        "none",
    ),
    "figshare": (
        "sourclip-2026-07-13-0162",
        "figshare_figshare_public_api_direct",
        "figshare_public_api",
        "none",
    ),
    "crossref": (
        "sourclip-2026-07-13-0202",
        "crossref_metadata_search_crossref_api_direct",
        "crossref_api",
        "none",
    ),
}

_EXPECTED_ORIGINS_AND_PATHS = {
    "arxiv": ("https", "export.arxiv.org", 443, ("/api/query",)),
    "pubmed": (
        "https",
        "eutils.ncbi.nlm.nih.gov",
        443,
        ("/entrez/eutils/esearch.fcgi", "/entrez/eutils/esummary.fcgi"),
    ),
    "semantic_scholar": (
        "https",
        "api.semanticscholar.org",
        443,
        ("/graph/v1/paper/search",),
    ),
    "zenodo": ("https", "zenodo.org", 443, ("/api/records",)),
    "openalex": ("https", "api.openalex.org", 443, ("/works",)),
    "osf": ("https", "api.osf.io", 443, ("/v2/preprints/",)),
    "figshare": ("https", "api.figshare.com", 443, ("/v2/articles/search",)),
    "crossref": ("https", "api.crossref.org", 443, ("/works",)),
}

_EXPECTED_PAGINATION_KEYS = {
    "openalex": "page",
    "semantic_scholar": "offset",
    "crossref": "offset",
    "arxiv": "start",
    "pubmed": "retstart",
    "zenodo": "page",
    "figshare": None,
    "osf": "page",
}


def _ledger_rows() -> dict[str, dict[str, object]]:
    payload = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    wanted_inventory_ids = {values[0] for values in _EXPECTED_LEDGER_ROUTES.values()}
    return {row["inventory_id"]: row for row in payload["rows"] if row["inventory_id"] in wanted_inventory_ids}


def test_foundation_registry_reconciles_all_eight_authoritative_rows() -> None:
    registry = foundation_registry()
    ledger_rows = _ledger_rows()

    assert len(registry.sources) == 8
    assert len(registry.routes) == 8
    assert len(registry.backends) == 8
    assert set(ledger_rows) == {values[0] for values in _EXPECTED_LEDGER_ROUTES.values()}

    for source_id, (inventory_id, route_id, backend_id, credential) in _EXPECTED_LEDGER_ROUTES.items():
        source = registry.get_source(source_id)
        route = registry.get_route(route_id)
        row = ledger_rows[inventory_id]
        route_candidate = row["route_candidates"][0]

        assert row["canonical_targets"] == [source_id]
        assert source.route_references == (SourceRouteReference(route_id, None),)
        assert route.route_id == route_candidate["route_candidate_id"] == route_id
        assert route.backend_id == route_candidate["planned_backend_id"] == backend_id
        assert route.route_kind.value == route_candidate["route_kind"] == RouteKind.DIRECT.value
        assert route.credential_requirement.value == route_candidate["credential_requirement"] == credential
        assert tuple(mode.value for mode in route.query_modes) == tuple(route_candidate["query_modes"])
        assert (
            route.source_constraint.value
            == route_candidate["source_constraint"]
            == SourceConstraint.NATIVE_CORPUS.value
        )
        assert source.route_references[0].source_predicate is route_candidate["source_constraint_predicate"]
        assert route.attribution_basis == route_candidate["attribution_basis"]


def test_foundation_route_policies_bind_exact_normalized_origins_and_paths() -> None:
    registry = foundation_registry()

    for source_id, expected in _EXPECTED_ORIGINS_AND_PATHS.items():
        route = registry.get_route(registry.get_source(source_id).route_references[0].route_id)
        origin = route.policy.origin
        actual = (origin.scheme, origin.host, origin.port, route.policy.paths)

        assert actual == expected
        assert route.policy.policy_digest
        assert route.query_modes == (QueryMode.STRUCTURED_QUERY,)


def test_foundation_registry_version_tracks_frozen_route_policy_content() -> None:
    assert foundation_registry().registry_version == "research-discovery-v2-foundation-2026-07-15"


def test_biorxiv_medrxiv_shadow_registry_is_strictly_additive_to_foundation() -> None:
    from tldw_Server_API.app.core.Research.discovery.biorxiv_medrxiv import (
        SHADOW_CATALOG_VERSION,
        biorxiv_medrxiv_shadow_registry,
    )

    foundation = foundation_registry()
    shadow = biorxiv_medrxiv_shadow_registry()

    assert shadow.sources[:8] == tuple(
        replace(source, catalog_version=SHADOW_CATALOG_VERSION) for source in foundation.sources
    )
    assert shadow.routes[:8] == foundation.routes
    assert shadow.backends[:8] == foundation.backends
    assert tuple(source.catalog_source_id for source in shadow.sources[8:]) == ("biorxiv", "medrxiv")


def test_foundation_routes_declare_exact_pagination_and_figshare_body_shape() -> None:
    registry = foundation_registry()

    actual = {
        source_id: registry.get_route(
            registry.get_source(source_id).route_references[0].route_id
        ).policy.pagination_query_key
        for source_id in _EXPECTED_PAGINATION_KEYS
    }
    figshare = registry.get_route("figshare_figshare_public_api_direct").policy

    assert actual == _EXPECTED_PAGINATION_KEYS
    assert figshare.methods == ("POST",)
    assert figshare.allowed_query_keys == ()
    assert figshare.pagination_json_body_key == "page"
    assert figshare.allowed_json_body_keys == (
        "search_for",
        "page",
        "page_size",
        "order",
        "order_direction",
    )
    assert figshare.integer_json_body_keys == ("page", "page_size")


def test_osf_policy_keeps_supported_plain_page_number_shape() -> None:
    policy = foundation_registry().get_route("open_science_framework_osf_api_direct").policy

    assert policy.pagination_query_key == "page"
    assert policy.allowed_query_keys == ("filter[title]", "page", "page[size]")
    assert "q" not in policy.allowed_query_keys
    assert "filter" not in policy.allowed_query_keys
    assert "page[number]" not in policy.allowed_query_keys


def test_zenodo_anonymous_route_caps_one_page_at_twenty_five_records() -> None:
    policy = foundation_registry().get_route("zenodo_zenodo_records_api_direct").policy

    assert policy.limits.max_pages == 1
    assert policy.limits.max_results == 25


def test_crossref_seed_alias_resolves_to_stable_product_identity() -> None:
    registry = foundation_registry()

    assert registry.resolve_source_id("crossref_metadata_search") == "crossref"
    assert registry.resolve_source_id("crossref-metadata-search") == "crossref"
    assert all(source.catalog_source_id != "crossref_metadata_search" for source in registry.sources)


def test_v2_registry_does_not_mutate_or_reinterpret_v1_catalog() -> None:
    before = [asdict(entry) for entry in default_source_catalog().list_sources()]

    registry = foundation_registry()

    after = [asdict(entry) for entry in default_source_catalog().list_sources()]
    assert after == before
    assert (
        next(entry for entry in after if entry["source_id"] == "openalex")["capabilities"]["requires_credentials"]
        is False
    )
    assert registry.get_route("openalex_openalex_api_direct").credential_requirement is CredentialRequirement.API_KEY


def test_foundation_readiness_is_explicit_fixture_or_synthetic_only() -> None:
    registry = foundation_registry()

    for mode in (ExecutionMode.OFFLINE_FIXTURE, ExecutionMode.SYNTHETIC):
        overlay = foundation_readiness(mode)
        by_route = {entry.route_id: entry for entry in overlay.routes}
        openalex_route_id = registry.get_source("openalex").route_references[0].route_id

        assert overlay.execution_mode is mode
        assert set(by_route) == {route.route_id for route in registry.routes}
        assert sum(entry.state is ReadinessState.READY for entry in overlay.routes) == 7
        assert by_route[openalex_route_id].state is ReadinessState.CREDENTIALED_OUT_OF_SCOPE
        assert by_route[openalex_route_id].credential_status is CredentialStatus.OUT_OF_SCOPE


def test_openalex_registry_declaration_has_no_secret_material_or_reference_interface() -> None:
    registry = foundation_registry()
    route = registry.get_route("openalex_openalex_api_direct")
    serialized = json.dumps(asdict(route), sort_keys=True)

    assert route.credential_requirement is CredentialRequirement.API_KEY
    assert "secret" not in serialized.casefold()
    assert "credential_ref" not in serialized.casefold()
    assert "api_key" not in route.policy.allowed_query_keys


def test_registry_rejects_dangling_route_and_backend_references() -> None:
    registry = foundation_registry()
    first_source = registry.sources[0]
    first_route = registry.routes[0]

    with pytest.raises(ValueError, match="unknown_route_reference"):
        DiscoveryRegistry(
            catalog_version=registry.catalog_version,
            registry_version=registry.registry_version,
            sources=(
                replace(
                    first_source,
                    route_references=(SourceRouteReference("missing_route", None),),
                ),
            ),
            routes=(first_route,),
            backends=registry.backends,
        )
    with pytest.raises(ValueError, match="unknown_backend_reference"):
        DiscoveryRegistry(
            catalog_version=registry.catalog_version,
            registry_version=registry.registry_version,
            sources=(first_source,),
            routes=(replace(first_route, backend_id="missing_backend"),),
            backends=registry.backends,
        )


@pytest.mark.parametrize("field_name", ["sources", "routes", "backends"])
def test_registry_rejects_untyped_nested_members(field_name: str) -> None:
    registry = foundation_registry()

    with pytest.raises(TypeError):
        replace(registry, **{field_name: (object(),)})


def test_registry_supports_multiple_routes_and_rejects_alias_collisions() -> None:
    registry = foundation_registry()
    source = registry.get_source("arxiv")
    primary = registry.get_route(source.route_references[0].route_id)
    fallback = replace(primary, route_id="arxiv_fixture_fallback", fallback_order=1)
    multi_route_source = replace(
        source,
        route_references=(
            SourceRouteReference(primary.route_id, None),
            SourceRouteReference(fallback.route_id, None),
        ),
    )
    multi_route = DiscoveryRegistry(
        catalog_version=registry.catalog_version,
        registry_version="synthetic-registry-v1",
        sources=(multi_route_source,),
        routes=(primary, fallback),
        backends=(registry.get_backend(primary.backend_id),),
    )

    assert [route.route_id for route in multi_route.routes_for_source("arxiv")] == [
        primary.route_id,
        fallback.route_id,
    ]

    colliding = SourceDefinition(
        catalog_source_id="second",
        display_name="Second",
        aliases=("arxiv",),
        categories=("preprints",),
        content_types=("works",),
        surfaces=("standalone_search",),
        route_references=(SourceRouteReference(primary.route_id, None),),
        site_hosts=(),
        priority=20,
        catalog_version=registry.catalog_version,
    )
    with pytest.raises(ValueError, match="source_alias_collision"):
        DiscoveryRegistry(
            catalog_version=registry.catalog_version,
            registry_version="bad-registry-v1",
            sources=(source, colliding),
            routes=(primary,),
            backends=(BackendDefinition(primary.backend_id, "arXiv API"),),
        )
