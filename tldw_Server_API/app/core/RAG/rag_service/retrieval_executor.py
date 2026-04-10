from __future__ import annotations

from typing import Any

from .evidence_models import RetrievedEvidence


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
    try:
        from .database_retrievers import MultiDatabaseRetriever as _MultiDatabaseRetriever
    except ImportError:
        _MultiDatabaseRetriever = None

    if _MultiDatabaseRetriever is not None and isinstance(retriever, _MultiDatabaseRetriever):
        documents = await retriever.retrieve_from_plan(
            retrieval_plan,
            config=retrieval_config,
            allowed_media_ids=allowed_media_ids,
            allowed_note_ids=allowed_note_ids,
        )
    else:
        documents = await retriever.retrieve(
            query=retrieval_plan.query,
            sources=list(retrieval_plan.sources),
            search_mode=retrieval_plan.search_mode,
            top_k=retrieval_plan.top_k,
            min_score=retrieval_plan.min_score,
            index_namespace=retrieval_plan.index_namespace,
            collection_names=dict(retrieval_plan.collection_names),
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
