"""Product-owned V2 source and route registry for the eight-source foundation."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CREDENTIALED_ROUTE_SKIP_REASON,
    AccessRoute,
    BackendDefinition,
    CredentialRequirement,
    CredentialStatus,
    ExactOrigin,
    ExecutionMode,
    QueryMode,
    ReadinessOverlay,
    ReadinessState,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    RouteReadiness,
    SourceConstraint,
    SourceDefinition,
    SourceRouteReference,
    canonical_policy_digest,
)

FOUNDATION_CATALOG_VERSION = "research-discovery-v2-foundation"
FOUNDATION_REGISTRY_VERSION = "research-discovery-v2-foundation-2026-07-15"
FOUNDATION_POLICY_VERSION = "research-discovery-route-policy-v2-foundation"
FOUNDATION_READINESS_VERSION = "research-discovery-readiness-v2-foundation"


@dataclass(frozen=True, slots=True)
class DiscoveryRegistry:
    """Immutable registry with validated source, route, backend, and alias links."""

    catalog_version: str
    registry_version: str
    sources: tuple[SourceDefinition, ...]
    routes: tuple[AccessRoute, ...]
    backends: tuple[BackendDefinition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.catalog_version, str) or not self.catalog_version:
            raise ValueError("invalid_registry_catalog_version")
        if not isinstance(self.registry_version, str) or not self.registry_version:
            raise ValueError("invalid_registry_version")
        for name in ("sources", "routes", "backends"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"registry_{name}_must_be_tuple")
        for name, values, value_type in (
            ("sources", self.sources, SourceDefinition),
            ("routes", self.routes, AccessRoute),
            ("backends", self.backends, BackendDefinition),
        ):
            if any(not isinstance(value, value_type) for value in values):
                raise TypeError(f"registry_{name}_must_be_typed")
        self._validate_unique_ids()
        self._validate_references()

    def get_source(self, source_id: str) -> SourceDefinition:
        """Return a source by its stable product ID."""
        canonical_id = self.resolve_source_id(source_id)
        return next(source for source in self.sources if source.catalog_source_id == canonical_id)

    def resolve_source_id(self, source_id: str) -> str:
        """Resolve a stable source ID or an explicit alias."""
        if not isinstance(source_id, str):
            raise KeyError(source_id)
        candidate = source_id.strip().casefold()
        for source in self.sources:
            if candidate == source.catalog_source_id or candidate in source.aliases:
                return source.catalog_source_id
        raise KeyError(source_id)

    def get_route(self, route_id: str) -> AccessRoute:
        """Return one access route by ID."""
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(route_id)

    def get_backend(self, backend_id: str) -> BackendDefinition:
        """Return one physical backend by ID."""
        for backend in self.backends:
            if backend.backend_id == backend_id:
                return backend
        raise KeyError(backend_id)

    def routes_for_source(self, source_id: str) -> tuple[AccessRoute, ...]:
        """Return a source's routes in declared fallback order."""
        source = self.get_source(source_id)
        return tuple(self.get_route(reference.route_id) for reference in source.route_references)

    def _validate_unique_ids(self) -> None:
        _ensure_unique(
            "catalog_source_id",
            tuple(source.catalog_source_id for source in self.sources),
        )
        _ensure_unique("route_id", tuple(route.route_id for route in self.routes))
        _ensure_unique("backend_id", tuple(backend.backend_id for backend in self.backends))

        names: dict[str, str] = {}
        for source in self.sources:
            for name in (source.catalog_source_id, *source.aliases):
                owner = names.get(name)
                if owner is not None and owner != source.catalog_source_id:
                    raise ValueError(f"source_alias_collision:{name}")
                names[name] = source.catalog_source_id

    def _validate_references(self) -> None:
        route_ids = {route.route_id for route in self.routes}
        backend_ids = {backend.backend_id for backend in self.backends}
        reference_counts = dict.fromkeys(route_ids, 0)

        for route in self.routes:
            if route.backend_id not in backend_ids:
                raise ValueError(f"unknown_backend_reference:{route.backend_id}")
            if route.policy.policy_digest != canonical_policy_digest(route.policy):
                raise ValueError(f"invalid_policy_digest:{route.route_id}")

        for source in self.sources:
            if source.catalog_version != self.catalog_version:
                raise ValueError(f"source_catalog_version_mismatch:{source.catalog_source_id}")
            previous_fallback_order = -1
            for reference in source.route_references:
                if reference.route_id not in route_ids:
                    raise ValueError(f"unknown_route_reference:{reference.route_id}")
                route = self.get_route(reference.route_id)
                reference_counts[route.route_id] += 1
                if route.fallback_order < previous_fallback_order:
                    raise ValueError(f"unordered_route_reference:{source.catalog_source_id}")
                previous_fallback_order = route.fallback_order
                if route.source_constraint is SourceConstraint.NATIVE_CORPUS:
                    if reference.source_predicate is not None:
                        raise ValueError(f"native_route_has_predicate:{route.route_id}")
                elif reference.source_predicate is None:
                    raise ValueError(f"constrained_route_requires_predicate:{route.route_id}")

        for route in self.routes:
            if reference_counts[route.route_id] == 0:
                raise ValueError(f"unreferenced_route:{route.route_id}")
            if route.source_constraint is SourceConstraint.NATIVE_CORPUS and reference_counts[route.route_id] != 1:
                raise ValueError(f"native_route_shared_across_sources:{route.route_id}")


def foundation_registry() -> DiscoveryRegistry:
    """Build the immutable eight-source V2 foundation registry."""
    routes = _foundation_routes()
    return DiscoveryRegistry(
        catalog_version=FOUNDATION_CATALOG_VERSION,
        registry_version=FOUNDATION_REGISTRY_VERSION,
        sources=_foundation_sources(),
        routes=routes,
        backends=tuple(
            BackendDefinition(backend_id, display_name)
            for backend_id, display_name in (
                ("openalex_api", "OpenAlex API"),
                ("semantic_scholar_graph_api", "Semantic Scholar Graph API"),
                ("crossref_api", "Crossref REST API"),
                ("arxiv_api", "arXiv Export API"),
                ("ncbi_eutils_pubmed", "NCBI E-utilities PubMed"),
                ("zenodo_records_api", "Zenodo Records API"),
                ("figshare_public_api", "Figshare Public API"),
                ("osf_api", "Open Science Framework API"),
            )
        ),
    )


def foundation_readiness(execution_mode: ExecutionMode) -> ReadinessOverlay:
    """Build explicit fixture or synthetic readiness for all foundation routes."""
    if not isinstance(execution_mode, ExecutionMode):
        raise TypeError("execution_mode_must_be_ExecutionMode")
    entries: list[RouteReadiness] = []
    for route in foundation_registry().routes:
        if route.credential_requirement is CredentialRequirement.NONE:
            entries.append(
                RouteReadiness(
                    route_id=route.route_id,
                    state=ReadinessState.READY,
                    credential_status=CredentialStatus.NOT_REQUIRED,
                    reason=f"{execution_mode.value}_ready",
                )
            )
        else:
            entries.append(
                RouteReadiness(
                    route_id=route.route_id,
                    state=ReadinessState.CREDENTIALED_OUT_OF_SCOPE,
                    credential_status=CredentialStatus.OUT_OF_SCOPE,
                    reason=CREDENTIALED_ROUTE_SKIP_REASON,
                )
            )
    return ReadinessOverlay(
        overlay_version=FOUNDATION_READINESS_VERSION,
        execution_mode=execution_mode,
        routes=tuple(entries),
    )


def _foundation_sources() -> tuple[SourceDefinition, ...]:
    return (
        _source(
            "openalex",
            "OpenAlex",
            10,
            "open_research_graph",
            ("works", "authors", "institutions", "venues"),
            "openalex_openalex_api_direct",
            aliases=("open_alex",),
            site_hosts=("openalex.org",),
        ),
        _source(
            "semantic_scholar",
            "Semantic Scholar",
            20,
            "open_research_graph",
            ("papers", "citations", "recommendations"),
            "semantic_scholar_semantic_scholar_graph_api_direct",
            aliases=("semantic-scholar",),
            site_hosts=("semanticscholar.org",),
        ),
        _source(
            "crossref",
            "Crossref",
            30,
            "open_research_graph",
            ("works", "publishers", "funders"),
            "crossref_metadata_search_crossref_api_direct",
            aliases=("crossref_metadata_search", "crossref-metadata-search"),
            site_hosts=("crossref.org", "doi.org"),
        ),
        _source(
            "arxiv",
            "arXiv",
            40,
            "preprints",
            ("preprints", "papers"),
            "arxiv_arxiv_api_direct",
            aliases=("ar_xiv",),
            site_hosts=("arxiv.org",),
        ),
        _source(
            "pubmed",
            "PubMed",
            50,
            "biomedical",
            ("papers", "abstracts", "biomedical_metadata"),
            "pubmed_ncbi_eutils_pubmed_direct",
            aliases=("pub_med",),
            site_hosts=("pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"),
        ),
        _source(
            "zenodo",
            "Zenodo",
            60,
            "repositories",
            ("datasets", "software", "papers"),
            "zenodo_zenodo_records_api_direct",
            site_hosts=("zenodo.org",),
        ),
        _source(
            "figshare",
            "Figshare",
            70,
            "repositories",
            ("datasets", "figures", "papers"),
            "figshare_figshare_public_api_direct",
            site_hosts=("figshare.com",),
        ),
        _source(
            "osf",
            "OSF",
            80,
            "repositories",
            ("projects", "registrations", "preprints"),
            "open_science_framework_osf_api_direct",
            aliases=("open_science_framework",),
            site_hosts=("osf.io",),
        ),
    )


def _foundation_routes() -> tuple[AccessRoute, ...]:
    return (
        _route(
            route_id="openalex_openalex_api_direct",
            backend_id="openalex_api",
            adapter_id="openalex_v2",
            host="api.openalex.org",
            methods=("GET",),
            paths=("/works",),
            query_keys=("search", "filter", "sort", "page", "per-page"),
            pagination_query_key="page",
            credential_requirement=CredentialRequirement.API_KEY,
        ),
        _route(
            route_id="semantic_scholar_semantic_scholar_graph_api_direct",
            backend_id="semantic_scholar_graph_api",
            adapter_id="semantic_scholar_v2",
            host="api.semanticscholar.org",
            methods=("GET",),
            paths=("/graph/v1/paper/search",),
            query_keys=("query", "offset", "limit", "fields", "fieldsOfStudy", "year"),
            pagination_query_key="offset",
        ),
        _route(
            route_id="crossref_metadata_search_crossref_api_direct",
            backend_id="crossref_api",
            adapter_id="crossref_v2",
            host="api.crossref.org",
            methods=("GET",),
            paths=("/works",),
            query_keys=("query", "query.title", "query.author", "filter", "offset", "rows", "select", "sort"),
            pagination_query_key="offset",
        ),
        _route(
            route_id="arxiv_arxiv_api_direct",
            backend_id="arxiv_api",
            adapter_id="arxiv_v2",
            host="export.arxiv.org",
            methods=("GET",),
            paths=("/api/query",),
            query_keys=("search_query", "start", "max_results", "sortBy", "sortOrder"),
            pagination_query_key="start",
        ),
        _route(
            route_id="pubmed_ncbi_eutils_pubmed_direct",
            backend_id="ncbi_eutils_pubmed",
            adapter_id="pubmed_v2",
            host="eutils.ncbi.nlm.nih.gov",
            methods=("GET",),
            paths=("/entrez/eutils/esearch.fcgi", "/entrez/eutils/esummary.fcgi"),
            query_keys=("db", "term", "retstart", "retmax", "retmode", "sort", "datetype", "mindate", "maxdate", "id"),
            pagination_query_key="retstart",
            max_physical_dispatches=2,
        ),
        _route(
            route_id="zenodo_zenodo_records_api_direct",
            backend_id="zenodo_records_api",
            adapter_id="zenodo_v2",
            host="zenodo.org",
            methods=("GET",),
            paths=("/api/records",),
            query_keys=("q", "page", "size", "sort"),
            pagination_query_key="page",
            max_results=25,
        ),
        _route(
            route_id="figshare_figshare_public_api_direct",
            backend_id="figshare_public_api",
            adapter_id="figshare_v2",
            host="api.figshare.com",
            methods=("POST",),
            paths=("/v2/articles/search",),
            query_keys=(),
            pagination_query_key=None,
            pagination_json_body_key="page",
            json_body_keys=("search_for", "page", "page_size", "order", "order_direction"),
            integer_json_body_keys=("page", "page_size"),
        ),
        _route(
            route_id="open_science_framework_osf_api_direct",
            backend_id="osf_api",
            adapter_id="osf_v2",
            host="api.osf.io",
            methods=("GET",),
            paths=("/v2/preprints/",),
            # The public preprints endpoint treats this as a title-substring filter.
            query_keys=("filter[title]", "page", "page[size]"),
            pagination_query_key="page",
        ),
    )


def _source(
    source_id: str,
    display_name: str,
    priority: int,
    category: str,
    content_types: tuple[str, ...],
    route_id: str,
    *,
    aliases: tuple[str, ...] = (),
    site_hosts: tuple[str, ...],
) -> SourceDefinition:
    return SourceDefinition(
        catalog_source_id=source_id,
        display_name=display_name,
        aliases=aliases,
        categories=(category,),
        content_types=content_types,
        surfaces=("standalone_search", "deep_research"),
        route_references=(SourceRouteReference(route_id, None),),
        site_hosts=site_hosts,
        priority=priority,
        catalog_version=FOUNDATION_CATALOG_VERSION,
    )


def _route(
    *,
    route_id: str,
    backend_id: str,
    adapter_id: str,
    host: str,
    methods: tuple[str, ...],
    paths: tuple[str, ...],
    query_keys: tuple[str, ...],
    pagination_query_key: str | None,
    pagination_json_body_key: str | None = None,
    json_body_keys: tuple[str, ...] = (),
    integer_json_body_keys: tuple[str, ...] = (),
    credential_requirement: CredentialRequirement = CredentialRequirement.NONE,
    max_physical_dispatches: int = 1,
    max_results: int = 100,
) -> AccessRoute:
    return AccessRoute(
        route_id=route_id,
        backend_id=backend_id,
        adapter_id=adapter_id,
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.STRUCTURED_QUERY,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_response",
        credential_requirement=credential_requirement,
        fallback_order=0,
        max_physical_dispatches=max_physical_dispatches,
        adapter_version="foundation-v2",
        policy=RoutePolicy(
            policy_version=FOUNDATION_POLICY_VERSION,
            origin=ExactOrigin("https", host, 443),
            methods=methods,
            paths=paths,
            allowed_query_keys=query_keys,
            pagination_query_key=pagination_query_key,
            pagination_json_body_key=pagination_json_body_key,
            allowed_json_body_keys=json_body_keys,
            integer_json_body_keys=integer_json_body_keys,
            limits=RouteLimits(
                max_pages=1,
                max_redirects=0,
                max_retries=0,
                timeout_ms=20_000,
                max_response_bytes=2_097_152,
                max_results=max_results,
            ),
        ),
    )


def _ensure_unique(name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate_{name}")
