from __future__ import annotations

from typing import Any

from ..models import AccessScope, ContextRequest, SearchRequest
from .search import DocsRetrievalService


class DocsContextBuilder:
    def __init__(self, retrieval: DocsRetrievalService) -> None:
        self.retrieval = retrieval

    def build(self, *, scope: AccessScope, request: ContextRequest) -> dict[str, Any]:
        max_chunks = max(int(request.max_chunks), 0)
        max_documents = max(int(request.max_documents), 0)
        max_characters = max(int(request.max_characters), 0)
        if max_chunks == 0 or max_documents == 0 or max_characters == 0:
            return self._empty_pack(request.query, max_characters)

        search = self.retrieval.search(
            scope=scope,
            request=SearchRequest(
                query=request.query,
                filters=request.filters,
                limit=max(max_chunks * 2, max_chunks),
                snippet_length=max_characters,
            ),
        )
        chunks: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        seen_documents: set[int] = set()
        used_characters = 0

        for result in search["results"]:
            if len(chunks) >= max_chunks:
                break
            document_id = int(result["document_id"])
            if document_id not in seen_documents and len(seen_documents) >= max_documents:
                continue

            text = str(result["snippet"])
            remaining = max_characters - used_characters
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining]
            if not text:
                continue

            used_characters += len(text)
            seen_documents.add(document_id)
            chunk = dict(result)
            chunk["text"] = text
            chunks.append(chunk)
            citations.append(
                {
                    "uri": result["uri"],
                    "citation": result["citation"],
                    "title": result["title"],
                }
            )

        return {
            "query": request.query,
            "chunks": chunks,
            "citations": citations,
            "omitted": max(0, len(search["results"]) - len(chunks)),
            "budget": {"max_characters": max_characters, "used_characters": used_characters},
            "warnings": search.get("warnings", []),
        }

    @staticmethod
    def _empty_pack(query: str, max_characters: int) -> dict[str, Any]:
        return {
            "query": query,
            "chunks": [],
            "citations": [],
            "omitted": 0,
            "budget": {"max_characters": max_characters, "used_characters": 0},
            "warnings": [],
        }
