from __future__ import annotations

from typing import Any

from ..models import AccessScope, SearchRequest
from ..store.sqlite import DocsCatalogStore


class DocsRetrievalService:
    """Read-side service for document search and catalog listing operations."""

    def __init__(self, store: DocsCatalogStore) -> None:
        self.store = store

    def search(self, *, scope: AccessScope, request: SearchRequest) -> dict[str, Any]:
        """Return a paged full-text search result set plus the total match count."""

        rows = self.store.search_chunks(
            scope=scope,
            query=request.query,
            limit=request.limit,
            offset=request.offset,
            filters=request.filters,
            snippet_length=request.snippet_length,
        )
        total = self.store.count_search_chunks(scope=scope, query=request.query, filters=request.filters)
        return {"results": rows, "count": total, "warnings": []}

    def get(self, *, scope: AccessScope, target: int | str, mode: str = "snippet") -> dict[str, Any]:
        """Return one document by id, URI, source, alias, or title."""

        return self.store.get_document(scope=scope, target=target, mode=mode)

    def list_documents(self, *, scope: AccessScope, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Return documents in the active access scope."""

        return {"documents": self.store.list_documents(scope=scope, limit=limit, offset=offset)}

    def list_collections(self, *, scope: AccessScope) -> dict[str, Any]:
        """Return collections and document counts in the active access scope."""

        return {"collections": self.store.list_collections(scope=scope)}

    def list_keywords(self, *, scope: AccessScope) -> dict[str, Any]:
        """Return keywords and document counts in the active access scope."""

        return {"keywords": self.store.list_keywords(scope=scope)}
