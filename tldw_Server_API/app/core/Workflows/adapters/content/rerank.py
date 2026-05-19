"""Reranking adapter.

This module includes the reranking adapter.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.content._config import RerankConfig

_RERANK_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


@registry.register(
    "rerank",
    category="content",
    description="Rerank search results",
    parallelizable=False,
    tags=["content", "ranking"],
    config_model=RerankConfig,
)
async def run_rerank_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Rerank documents using various scoring strategies.

    Config:
      - query: str (templated) - the search query
      - documents: Optional[List[Dict]] - from last.documents or explicit
      - strategy: str = "flashrank" - reranking strategy
        Options: "flashrank", "cross_encoder", "llm_scoring", "diversity",
                 "multi_criteria", "hybrid", "llama_cpp", "two_tier"
      - top_k: int = 10 - number of documents to return
      - api_name: Optional[str] - for LLM-based strategies
    Output:
      - {"documents": [{content, score, metadata, original_index}], "count": int, "strategy": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # Template rendering for query
    query_t = str(config.get("query") or "").strip()
    if query_t:
        query = apply_template_to_string(query_t, context) or query_t
    else:
        # Try to get from last.query or last.combined
        query = None
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                query = str(last.get("query") or last.get("combined") or last.get("text") or "")
        except (AttributeError, TypeError, ValueError):
            pass
    query = query or ""

    if not query.strip():
        return {"error": "missing_query", "documents": [], "count": 0}

    # Get documents
    documents_raw = config.get("documents")
    documents: list[dict[str, Any]] = []

    if documents_raw:
        # Template if it's a string reference
        if isinstance(documents_raw, str):
            rendered = apply_template_to_string(documents_raw, context)
            try:
                documents = json.loads(rendered) if rendered else []
            except (TypeError, ValueError, json.JSONDecodeError):
                documents = []
        elif isinstance(documents_raw, list):
            documents = documents_raw
    else:
        # Try to get from last.documents
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                docs = last.get("documents") or last.get("results") or []
                if isinstance(docs, list):
                    documents = docs
        except (AttributeError, TypeError, ValueError):
            pass

    if not documents:
        return {"error": "missing_documents", "documents": [], "count": 0}

    strategy = str(config.get("strategy") or "flashrank").strip().lower()
    top_k = int(config.get("top_k") or 10)
    top_k = max(1, min(top_k, 100))

    # Test mode simulation
    if is_test_mode():
        # Simulate reranking by adding scores
        reranked = []
        for i, doc in enumerate(documents[:top_k]):
            score = 1.0 - (i * 0.1)  # Decreasing score
            score = max(0.1, min(1.0, score))
            reranked.append({
                "content": doc.get("content") or doc.get("text") or str(doc),
                "score": score,
                "original_index": i,
                "metadata": doc.get("metadata") or {},
            })
        return {
            "documents": reranked,
            "count": len(reranked),
            "strategy": strategy,
            "query": query,
            "simulated": True,
        }

    try:
        from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
            RerankingConfig,
            RerankingStrategy,
            ScoredDocument,
            create_reranker,
        )
        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

        # Map strategy string to enum
        strategy_enum_map = {
            "flashrank": RerankingStrategy.FLASHRANK,
            "cross_encoder": RerankingStrategy.CROSS_ENCODER,
            "llm_scoring": RerankingStrategy.LLM_SCORING,
            "diversity": RerankingStrategy.DIVERSITY,
            "multi_criteria": RerankingStrategy.MULTI_CRITERIA,
            "hybrid": RerankingStrategy.HYBRID,
            "llama_cpp": RerankingStrategy.LLAMA_CPP,
            "two_tier": RerankingStrategy.TWO_TIER,
        }

        if strategy not in strategy_enum_map:
            return {"error": f"invalid_strategy:{strategy}", "documents": [], "count": 0}

        strategy_enum = strategy_enum_map[strategy]

        # Build config
        rerank_config = RerankingConfig(
            strategy=strategy_enum,
            top_k=top_k,
            model_name=config.get("model_name"),
        )

        # Convert input documents to Document objects
        doc_objects: list[Document] = []
        for i, doc in enumerate(documents):
            content = doc.get("content") or doc.get("text") or str(doc)
            doc_obj = Document(
                id=doc.get("id") or f"doc_{i}",
                content=content,
                metadata=doc.get("metadata") or {},
                source=DataSource.WEB_CONTENT,
                score=float(doc.get("score") or 0.5),
            )
            doc_objects.append(doc_obj)

        # Create reranker and run
        reranker = create_reranker(strategy_enum, rerank_config)
        scored_docs: list[ScoredDocument] = await reranker.rerank(query, doc_objects)

        # Convert results
        output_docs = []
        for sd in scored_docs:
            output_docs.append({
                "content": sd.document.content,
                "score": sd.rerank_score,
                "original_index": doc_objects.index(sd.document) if sd.document in doc_objects else -1,
                "metadata": sd.document.metadata,
                "original_score": sd.original_score,
                "relevance_score": sd.relevance_score,
            })

        return {
            "documents": output_docs,
            "count": len(output_docs),
            "strategy": strategy,
            "query": query,
        }

    except _RERANK_NONCRITICAL_EXCEPTIONS:
        logger.exception("Rerank adapter error")
        return {"error": "rerank_error"}
