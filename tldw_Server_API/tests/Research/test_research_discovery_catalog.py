def test_catalog_lists_first_slice_sources_with_capabilities():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog()
    source_ids = {source.source_id for source in catalog.list_sources()}

    assert {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    }.issubset(source_ids)
    assert catalog.get_source("openalex").capabilities.searchable is True
    assert catalog.get_source("openalex").capabilities.fallback_search_allowed is False
    assert catalog.catalog_version


def test_catalog_resolves_category_and_rejects_over_cap_selection():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog(max_selected_sources=2)

    resolved, error = catalog.resolve_selection(source_ids=[], categories=["open_research_graph"])

    assert resolved == []
    assert error is not None
    assert error.code == "source_selection_over_cap"
    assert error.selected_count > error.limit


def test_catalog_dedupes_explicit_and_category_selected_sources():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog(max_selected_sources=10)
    resolved, error = catalog.resolve_selection(
        source_ids=["openalex"],
        categories=["open_research_graph"],
    )

    assert error is None
    assert resolved[0].source_id == "openalex"
    assert len({entry.source_id for entry in resolved}) == len(resolved)
