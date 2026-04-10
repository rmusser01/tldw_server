from __future__ import annotations

import inspect
from typing import Any

from .evidence_models import RetrievedEvidence
from .types import DataSource


async def execute_retrieval_phase(
    *,
    resolved_request: Any,
    retrieval_plan: Any,
    retriever: Any,
    retrieval_config: Any | None = None,
    allowed_media_ids: list[int] | None = None,
    allowed_note_ids: list[str] | None = None,
) -> RetrievedEvidence:
    """Execute a normalized retrieval phase and package canonical evidence."""
    def _legacy_sources() -> list[DataSource]:
        sources: list[DataSource] = []
        for source in getattr(retrieval_plan, "sources", ()) or ():
            if isinstance(source, DataSource):
                sources.append(source)
                continue
            try:
                sources.append(DataSource(str(source)))
            except (TypeError, ValueError):
                continue
        return sources

    retrieval_kwargs: dict[str, Any] = {
        "config": retrieval_config,
        "allowed_media_ids": allowed_media_ids,
        "allowed_note_ids": allowed_note_ids,
    }
    retrieve_from_plan = getattr(retriever, "retrieve_from_plan", None)
    if callable(retrieve_from_plan) and inspect.iscoroutinefunction(retrieve_from_plan):
        documents = await retriever.retrieve_from_plan(
            retrieval_plan,
            **retrieval_kwargs,
        )
    else:
        documents = await retriever.retrieve(
            query=retrieval_plan.query,
            sources=_legacy_sources(),
            search_mode=retrieval_plan.search_mode,
            top_k=retrieval_plan.top_k,
            min_score=retrieval_plan.min_score,
            index_namespace=retrieval_plan.index_namespace,
            collection_names=dict(retrieval_plan.collection_names),
            retrieval_plan=retrieval_plan,
            **retrieval_kwargs,
        )

    return RetrievedEvidence(
        documents=list(documents),
        metadata={
            "resolved_request": {
                "query": resolved_request.query,
                "user_id": resolved_request.user_id,
            },
            "retrieval_plan": {
                "query": retrieval_plan.query,
                "sources": list(retrieval_plan.sources),
                "search_mode": retrieval_plan.search_mode,
                "top_k": retrieval_plan.top_k,
                "index_namespace": retrieval_plan.index_namespace,
            },
        },
    )
