from __future__ import annotations

from typing import Any

from ..models import AccessScope, SearchRequest
from ..store.sqlite import DocsCatalogStore


class DocsRetrievalService:
    def __init__(self, store: DocsCatalogStore) -> None:
        self.store = store

    def search(self, *, scope: AccessScope, request: SearchRequest) -> dict[str, Any]:
        rows = self.store.search_chunks(
            scope=scope,
            query=request.query,
            limit=request.limit,
            offset=request.offset,
            filters=request.filters,
            snippet_length=request.snippet_length,
        )
        return {"results": rows, "count": len(rows), "warnings": []}

    def get(self, *, scope: AccessScope, target: int | str, mode: str = "snippet") -> dict[str, Any]:
        return self.store.get_document(scope=scope, target=target, mode=mode)

    def list_documents(self, *, scope: AccessScope, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return {"documents": self.store.list_documents(scope=scope, limit=limit, offset=offset)}

    def list_collections(self, *, scope: AccessScope) -> dict[str, Any]:
        return {"collections": self.store.list_collections(scope=scope)}

    def list_keywords(self, *, scope: AccessScope) -> dict[str, Any]:
        return {"keywords": self.store.list_keywords(scope=scope)}
