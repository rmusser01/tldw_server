"""Default source catalog for research discovery."""

from __future__ import annotations

from collections.abc import Sequence

from .models import ResearchSourceCatalogEntry, SourceCapabilities, SourceSelectionError


CATALOG_VERSION = "research-discovery-v1"
DEFAULT_MAX_SELECTED_SOURCES = 8


class ResearchSourceCatalog:
    """In-memory catalog for resolving research discovery source selections."""

    def __init__(
        self,
        entries: Sequence[ResearchSourceCatalogEntry],
        *,
        max_selected_sources: int = DEFAULT_MAX_SELECTED_SOURCES,
        catalog_version: str = CATALOG_VERSION,
    ) -> None:
        self.catalog_version = catalog_version
        self.max_selected_sources = max_selected_sources
        self._ensure_unique_source_ids(entries)
        self._entries = tuple(sorted(entries, key=_source_sort_key))
        self._entries_by_id = {entry.source_id: entry for entry in self._entries}
        self._categories = frozenset(entry.category for entry in self._entries)

    def list_sources(self) -> list[ResearchSourceCatalogEntry]:
        """Return all catalog entries in deterministic priority order."""
        return list(self._entries)

    def get_source(self, source_id: str) -> ResearchSourceCatalogEntry:
        """Return one catalog entry by source id."""
        return self._entries_by_id[source_id]

    def resolve_selection(
        self,
        source_ids: list[str],
        categories: list[str],
    ) -> tuple[list[ResearchSourceCatalogEntry], SourceSelectionError | None]:
        """Expand source/category selections, dedupe, and enforce the source cap."""
        unknown_source_ids = [source_id for source_id in source_ids if source_id not in self._entries_by_id]
        if unknown_source_ids:
            return [], SourceSelectionError(
                code="unknown_source",
                message=("Unknown research discovery source id(s): " f"{', '.join(unknown_source_ids)}."),
                selected_count=len(unknown_source_ids),
                limit=self.max_selected_sources,
            )

        unknown_categories = [category for category in categories if category not in self._categories]
        if unknown_categories:
            return [], SourceSelectionError(
                code="unknown_category",
                message=("Unknown research discovery source category/categories: " f"{', '.join(unknown_categories)}."),
                selected_count=len(unknown_categories),
                limit=self.max_selected_sources,
            )

        selected_by_id: dict[str, ResearchSourceCatalogEntry] = {}

        for source_id in source_ids:
            selected_by_id[source_id] = self.get_source(source_id)

        selected_categories = set(categories)
        for entry in self._entries:
            if entry.category in selected_categories:
                selected_by_id[entry.source_id] = entry

        selected_entries = sorted(selected_by_id.values(), key=_source_sort_key)
        if len(selected_entries) > self.max_selected_sources:
            return [], SourceSelectionError(
                code="source_selection_over_cap",
                message=(
                    "Source selection expands to "
                    f"{len(selected_entries)} sources, exceeding the limit of "
                    f"{self.max_selected_sources}."
                ),
                selected_count=len(selected_entries),
                limit=self.max_selected_sources,
            )

        return selected_entries, None

    @staticmethod
    def _ensure_unique_source_ids(entries: Sequence[ResearchSourceCatalogEntry]) -> None:
        seen_source_ids: set[str] = set()
        for entry in entries:
            if entry.source_id in seen_source_ids:
                raise ValueError(f"duplicate_source_id:{entry.source_id}")
            seen_source_ids.add(entry.source_id)


def default_source_catalog(
    *,
    max_selected_sources: int = DEFAULT_MAX_SELECTED_SOURCES,
) -> ResearchSourceCatalog:
    """Build the default research discovery source catalog."""
    return ResearchSourceCatalog(
        _default_entries(),
        max_selected_sources=max_selected_sources,
    )


def _default_entries() -> tuple[ResearchSourceCatalogEntry, ...]:
    return (
        _entry(
            source_id="openalex",
            display_name="OpenAlex",
            category="open_research_graph",
            content_types=("works", "authors", "institutions", "venues"),
            access_level="open_metadata",
            priority=10,
            provider_adapter="openalex",
            site_hosts=("openalex.org",),
            trust_notes="Open scholarly graph with broad metadata coverage.",
            full_text_resolvable=False,
            ingestable=False,
        ),
        _entry(
            source_id="semantic_scholar",
            display_name="Semantic Scholar",
            category="open_research_graph",
            content_types=("papers", "citations", "recommendations"),
            access_level="open_metadata",
            priority=20,
            provider_adapter="semantic_scholar",
            site_hosts=("semanticscholar.org",),
            trust_notes="Open metadata and citation graph with API rate limits.",
            full_text_resolvable=False,
            ingestable=False,
        ),
        _entry(
            source_id="crossref",
            display_name="Crossref",
            category="open_research_graph",
            content_types=("works", "publishers", "funders"),
            access_level="open_metadata",
            priority=30,
            provider_adapter="crossref",
            site_hosts=("crossref.org", "doi.org"),
            trust_notes="Publisher DOI metadata registry.",
            full_text_resolvable=False,
            ingestable=False,
        ),
        _entry(
            source_id="arxiv",
            display_name="arXiv",
            category="preprints",
            subcategory="general_preprints",
            content_types=("preprints", "papers"),
            access_level="open_full_text",
            priority=40,
            provider_adapter="arxiv",
            site_hosts=("arxiv.org",),
            trust_notes="Open preprint repository with direct full-text access.",
            full_text_resolvable=True,
            ingestable=True,
        ),
        _entry(
            source_id="pubmed",
            display_name="PubMed",
            category="biomedical",
            content_types=("papers", "abstracts", "biomedical_metadata"),
            access_level="open_metadata",
            priority=50,
            provider_adapter="pubmed",
            site_hosts=("pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"),
            trust_notes="Biomedical literature metadata from NCBI.",
            full_text_resolvable=True,
            ingestable=False,
        ),
        _entry(
            source_id="zenodo",
            display_name="Zenodo",
            category="repositories",
            content_types=("datasets", "software", "papers"),
            access_level="open_repository",
            priority=60,
            provider_adapter="zenodo",
            site_hosts=("zenodo.org",),
            trust_notes="Open research repository operated by CERN.",
            full_text_resolvable=True,
            ingestable=True,
        ),
        _entry(
            source_id="figshare",
            display_name="Figshare",
            category="repositories",
            content_types=("datasets", "figures", "papers"),
            access_level="open_repository",
            priority=70,
            provider_adapter="figshare",
            site_hosts=("figshare.com",),
            trust_notes="Research repository for datasets, figures, and papers.",
            full_text_resolvable=True,
            ingestable=True,
        ),
        _entry(
            source_id="osf",
            display_name="OSF",
            category="repositories",
            content_types=("projects", "registrations", "preprints"),
            access_level="open_repository",
            priority=80,
            provider_adapter="osf",
            site_hosts=("osf.io",),
            trust_notes="Open Science Framework project and registration metadata.",
            full_text_resolvable=True,
            ingestable=True,
        ),
    )


def _entry(
    *,
    source_id: str,
    display_name: str,
    category: str,
    content_types: tuple[str, ...],
    access_level: str,
    priority: int,
    provider_adapter: str,
    site_hosts: tuple[str, ...],
    trust_notes: str,
    full_text_resolvable: bool,
    ingestable: bool,
    subcategory: str | None = None,
) -> ResearchSourceCatalogEntry:
    return ResearchSourceCatalogEntry(
        source_id=source_id,
        display_name=display_name,
        category=category,
        subcategory=subcategory,
        content_types=content_types,
        access_level=access_level,
        enabled=True,
        configured=True,
        default_discovery_mode="api",
        fallback_enabled=False,
        priority=priority,
        provider_adapter=provider_adapter,
        site_hosts=site_hosts,
        trust_notes=trust_notes,
        capabilities=SourceCapabilities(
            searchable=True,
            full_text_resolvable=full_text_resolvable,
            ingestable=ingestable,
            requires_credentials=False,
            fallback_search_allowed=False,
            rate_limited=True,
        ),
        catalog_version=CATALOG_VERSION,
    )


def _source_sort_key(entry: ResearchSourceCatalogEntry) -> tuple[int, str]:
    return entry.priority, entry.source_id
