from __future__ import annotations

from pathlib import Path

from mcp_unified.docs.models import AccessScope, ContextRequest, SearchFilters, SearchRequest
from mcp_unified.docs.retrieval.context import DocsContextBuilder
from mcp_unified.docs.retrieval.search import DocsRetrievalService
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def _seed_store(tmp_path: Path) -> tuple[DocsCatalogStore, AccessScope, AccessScope]:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")
    store.upsert_document(
        scope=scope_a,
        title="SQLite Guide",
        document_type="markdown",
        canonical_uri="file:///docs/sqlite.md",
        source_path="/docs/sqlite.md",
        source_url=None,
        text="SQLite FTS5 supports local retrieval. Agents need citations.",
        sections=[],
        chunks=[
            {
                "text": "SQLite FTS5 supports local retrieval. Agents need citations.",
                "citation": "sqlite.md:1",
            }
        ],
        keywords=("database", "fts"),
        collection_names=("Reference",),
        metadata={"package": "sqlite", "version": "3"},
    )
    store.upsert_document(
        scope=scope_b,
        title="Other Guide",
        document_type="markdown",
        canonical_uri="file:///docs/other.md",
        source_path="/docs/other.md",
        source_url=None,
        text="SQLite FTS5 supports local retrieval in another scope.",
        sections=[],
        chunks=[{"text": "SQLite FTS5 supports local retrieval in another scope.", "citation": "other.md:1"}],
        keywords=("database",),
        collection_names=("Other",),
        metadata={"package": "sqlite", "version": "3"},
    )
    return store, scope_a, scope_b


def test_search_filters_by_collection_keyword_and_scope(tmp_path: Path) -> None:
    store, scope_a, scope_b = _seed_store(tmp_path)
    service = DocsRetrievalService(store)

    response = service.search(
        scope=scope_a,
        request=SearchRequest(
            query="retrieval",
            filters=SearchFilters(collection="Reference", keywords=("fts",)),
        ),
    )
    denied_response = service.search(
        scope=scope_b,
        request=SearchRequest(
            query="retrieval",
            filters=SearchFilters(collection="Reference", keywords=("fts",)),
        ),
    )

    assert response["count"] == 1  # nosec B101
    assert response["results"][0]["title"] == "SQLite Guide"  # nosec B101
    assert response["results"][0]["citation"] == "sqlite.md:1"  # nosec B101
    assert denied_response == {"results": [], "count": 0, "warnings": []}  # nosec B101


def test_context_pack_respects_character_budget(tmp_path: Path) -> None:
    store, scope, _other_scope = _seed_store(tmp_path)
    builder = DocsContextBuilder(DocsRetrievalService(store))

    pack = builder.build(
        scope=scope,
        request=ContextRequest(query="SQLite", max_chunks=2, max_characters=40),
    )

    assert pack["budget"]["max_characters"] == 40  # nosec B101
    assert pack["budget"]["used_characters"] <= 40  # nosec B101
    assert len(pack["chunks"]) == 1  # nosec B101
    assert len(pack["chunks"][0]["text"]) <= 40  # nosec B101
    assert pack["citations"][0]["uri"] == "file:///docs/sqlite.md"  # nosec B101


def test_list_collections_keywords_and_documents(tmp_path: Path) -> None:
    store, scope, _other_scope = _seed_store(tmp_path)
    service = DocsRetrievalService(store)

    collections = service.list_collections(scope=scope)
    keywords = service.list_keywords(scope=scope)
    documents = service.list_documents(scope=scope)

    assert collections["collections"][0]["name"] == "Reference"  # nosec B101
    assert {item["keyword"] for item in keywords["keywords"]} == {"database", "fts"}  # nosec B101
    assert documents["documents"][0]["title"] == "SQLite Guide"  # nosec B101
