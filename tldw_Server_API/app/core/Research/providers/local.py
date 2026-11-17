"""Local corpus provider for deep research collection."""

from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MultiDatabaseRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.testing import is_test_mode

_SOURCE_TO_DB_KEY = {
    "media_db": "media_db",
    "notes": "notes_db",
    "prompts": "prompts_db",
    "kanban": "kanban_db",
}

_SOURCE_TO_DATA_SOURCE = {
    "media_db": DataSource.MEDIA_DB,
    "notes": DataSource.NOTES,
    "prompts": DataSource.PROMPTS,
    "kanban": DataSource.KANBAN,
}


def _build_query(query: str, focus_area: str) -> str:
    return " ".join(part.strip() for part in (query, focus_area) if part and part.strip())


def _truncate_text(text: str, *, max_len: int = 400) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


class LocalResearchProvider:
    """Retrieve evidence from the user's local corpus."""

    def __init__(self, *, retriever_cls: type[MultiDatabaseRetriever] = MultiDatabaseRetriever) -> None:
        self._retriever_cls = retriever_cls

    async def search(
        self,
        *,
        focus_area: str,
        query: str,
        owner_user_id: str,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if is_test_mode():
            return [
                {
                    "id": f"local-{owner_user_id}-{focus_area}",
                    "title": f"Local evidence for {focus_area}",
                    "url": None,
                    "snippet": f"Simulated local corpus note about {query}",
                    "content": f"Simulated local corpus note about {query}",
                    "provider": "local_corpus",
                    "metadata": {"source": "media_db"},
                }
            ]

        sources = [str(item).strip() for item in config.get("sources", ["media_db"]) if str(item).strip()]
        db_paths: dict[str, str] = {}
        if "media_db" in sources:
            db_paths["media_db"] = str(DatabasePaths.get_media_db_path(owner_user_id))
        if "notes" in sources:
            db_paths["notes_db"] = str(DatabasePaths.get_chacha_db_path(owner_user_id))
        if "prompts" in sources:
            db_paths["prompts_db"] = str(DatabasePaths.get_prompts_db_path(owner_user_id))
        if "kanban" in sources:
            db_paths["kanban_db"] = str(DatabasePaths.get_kanban_db_path(owner_user_id))

        if not db_paths:
            return []

        use_vector = config.get("use_vector", True)
        if not isinstance(use_vector, bool):
            raise ValueError("local use_vector must be a boolean")
        retrieval_config = RetrievalConfig(
            max_results=int(config.get("top_k", 5)),
            use_fts=True,
            use_vector=use_vector,
        )
        retriever = self._retriever_cls(
            db_paths=db_paths,
            user_id=str(owner_user_id),
            media_config=retrieval_config,
        )
        try:
            documents = await retriever.retrieve(
                _build_query(query, focus_area),
                sources=[_SOURCE_TO_DATA_SOURCE[name] for name in sources if name in _SOURCE_TO_DATA_SOURCE],
                config=retrieval_config,
            )
        finally:
            close_fn = getattr(retriever, "close", None)
            if callable(close_fn):
                close_fn()

        return [self._normalize_document(document) for document in documents]

    @staticmethod
    def _normalize_document(document: Document) -> dict[str, Any]:
        metadata = to_jsonable_python(dict(document.metadata or {}))
        title = str(
            metadata.get("title")
            or metadata.get("document_title")
            or document.section_title
            or document.id
        ).strip()
        content = str(document.content or "").strip()
        return {
            "id": str(document.id),
            "title": title,
            "url": metadata.get("url"),
            "snippet": _truncate_text(content),
            "content": content,
            "provider": "local_corpus",
            "score": float(document.score),
            "metadata": metadata,
            "source": (
                document.source.value
                if isinstance(document.source, DataSource)
                else str(document.source)
            ),
        }


__all__ = ["LocalResearchProvider"]
