"""Query processing and optimization adapters.

This module includes adapters for query operations:
- query_rewrite: Rewrite search queries for better retrieval
- query_expand: Expand queries with synonyms and related terms
- hyde_generate: Generate hypothetical documents (HyDE)
- semantic_cache_check: Check semantic cache for similar queries
- search_aggregate: Aggregate and deduplicate search results
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.rag._config import (
    HyDEGenerateConfig,
    QueryExpandConfig,
    QueryRewriteConfig,
    SearchAggregateConfig,
    SemanticCacheCheckConfig,
)

_QUERY_CONTEXT_EXCEPTIONS = (AttributeError, TypeError, ValueError)
_QUERY_STRATEGY_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_QUERY_ADAPTER_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_QUERY_JSON_PARSE_EXCEPTIONS = (TypeError, ValueError, json.JSONDecodeError)


def _get_workflow_chroma_manager():
    """Resolve the default ChromaDB manager for workflow adapters."""
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import get_default_chroma_manager

    return get_default_chroma_manager()


def _get_semantic_cache_collection(collection_name: str) -> tuple[Any, Any]:
    """Resolve semantic-cache collection and return (manager, collection)."""
    manager = _get_workflow_chroma_manager()
    if not manager:
        return None, None
    collection = manager.get_or_create_collection(collection_name=collection_name)
    return manager, collection


def _build_semantic_cache_query_embedding(query: str, manager: Any) -> list[float]:
    """Build query embedding for semantic cache lookup."""
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import create_embedding

    model_override = getattr(manager, "default_embedding_model_id", None)
    user_cfg = getattr(manager, "user_embedding_config", None)
    if not isinstance(user_cfg, dict):
        raise ValueError("Chroma manager is missing embedding configuration.")
    return create_embedding(
        text=query,
        user_app_config=user_cfg,
        model_id_override=model_override,
    )


@registry.register(
    "query_rewrite",
    category="rag",
    description="Rewrite search queries",
    parallelizable=True,
    tags=["rag", "query"],
    config_model=QueryRewriteConfig,
)
async def run_query_rewrite_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a search query for better retrieval results.

    Config:
      - query: str (required, templated) - Original query to rewrite
      - strategy: Literal["expand", "clarify", "simplify", "all"] = "all"
      - provider: str - LLM provider for rewriting
      - model: str - Model to use
      - max_rewrites: int = 3 - Maximum number of rewritten queries
    Output:
      - {"original_query": str, "rewritten_queries": [str], "strategy": str}
    """
    # Check cancellation
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = apply_template_to_string(query, context) or query
    query = str(query).strip()

    if not query:
        return {"error": "missing_query", "original_query": "", "rewritten_queries": []}

    strategy = str(config.get("strategy", "all")).lower()
    max_rewrites = int(config.get("max_rewrites", 3))
    provider = config.get("provider")
    model = config.get("model")

    # Build rewrite prompt based on strategy
    strategy_prompts = {
        "expand": "Expand the query with synonyms, related terms, and alternative phrasings.",
        "clarify": "Clarify ambiguous terms and add context to make the query more specific.",
        "simplify": "Simplify the query to its core concepts, removing unnecessary words.",
        "all": "Generate variations using expansion, clarification, and simplification techniques.",
    }

    system_prompt = f"""You are a search query optimizer. {strategy_prompts.get(strategy, strategy_prompts['all'])}

Return exactly {max_rewrites} rewritten queries, one per line. No numbering, no explanations, just the queries."""

    user_prompt = f"Rewrite this search query:\n{query}"

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        messages = [{"role": "user", "content": user_prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=provider,
            model=model,
            system_message=system_prompt,
            max_tokens=500,
            temperature=0.7,
        )

        # Extract text from response
        text = extract_openai_content(response) or ""
        rewrites = [line.strip() for line in text.strip().split("\n") if line.strip()][:max_rewrites]

        return {
            "original_query": query,
            "rewritten_queries": rewrites,
            "strategy": strategy,
        }

    except _QUERY_ADAPTER_EXCEPTIONS:
        logger.exception("Query rewrite adapter error")
        return {"error": "query_rewrite_error", "original_query": query, "rewritten_queries": []}


@registry.register(
    "query_expand",
    category="rag",
    description="Expand search queries",
    parallelizable=True,
    tags=["rag", "query"],
    config_model=QueryExpandConfig,
)
async def run_query_expand_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Expand search queries using multiple strategies.

    Config:
      - query: str (templated) - Query to expand
      - strategies: List[str] = ["synonym"] - Strategies to use
        Options: "acronym", "synonym", "domain", "entity", "multi_query", "hybrid"
      - max_expansions: int = 5 - Max expansions per strategy
      - domain_context: Optional[str] (templated) - For domain strategy
      - api_name: Optional[str] - For LLM-based strategies
    Output:
      - {"original": str, "variations": [str], "synonyms": [str],
         "keywords": [str], "entities": [str], "combined": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # Template rendering for query
    query_t = str(config.get("query") or "").strip()
    if query_t:
        query = apply_template_to_string(query_t, context) or query_t
    else:
        # Try to get from last.text or last.query
        query = None
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                query = str(last.get("query") or last.get("text") or "")
        except _QUERY_CONTEXT_EXCEPTIONS:
            pass
    query = query or ""

    if not query.strip():
        return {"error": "missing_query", "original": "", "variations": [], "combined": ""}

    strategies = config.get("strategies") or ["synonym"]
    if isinstance(strategies, str):
        strategies = [s.strip() for s in strategies.split(",") if s.strip()]
    strategies = [s.lower().strip() for s in strategies]

    max_expansions = int(config.get("max_expansions") or 5)
    max_expansions = max(1, min(max_expansions, 20))

    domain_context = None
    domain_context_t = config.get("domain_context")
    if domain_context_t:
        domain_context = apply_template_to_string(str(domain_context_t), context) or str(domain_context_t)

    # Test mode simulation
    if is_test_mode():
        # Simulate query expansion
        simulated_variations = [
            f"{query} (expanded)",
            f"alternative {query}",
            f"{query} definition",
        ][:max_expansions]
        words = query.lower().split()
        return {
            "original": query,
            "variations": simulated_variations,
            "synonyms": {w: [f"{w}_syn"] for w in words[:3]},
            "keywords": words,
            "entities": [w.capitalize() for w in words if len(w) > 3][:2],
            "combined": f"{query} {simulated_variations[0] if simulated_variations else ''}".strip(),
            "strategies_used": strategies,
            "simulated": True,
        }

    try:
        from tldw_Server_API.app.core.RAG.rag_service.query_expansion import (
            AcronymExpansion,
            DomainExpansion,
            EntityExpansion,
            HybridQueryExpansion,
            MultiQueryGeneration,
            SynonymExpansion,
        )

        # Build strategy instances
        strategy_map = {
            "synonym": SynonymExpansion,
            "multi_query": MultiQueryGeneration,
            "acronym": AcronymExpansion,
            "domain": lambda: DomainExpansion(custom_terms={domain_context: []} if domain_context else None),
            "entity": EntityExpansion,
        }

        all_variations: list[str] = []
        all_synonyms: dict[str, list[str]] = {}
        all_keywords: list[str] = []
        all_entities: list[str] = []

        # If hybrid, use the combined strategy
        if "hybrid" in strategies:
            expander = HybridQueryExpansion()
            result = await expander.expand(query)
            all_variations.extend(result.variations[:max_expansions])
            all_synonyms.update(result.synonyms)
            all_keywords.extend(result.keywords)
            all_entities.extend(result.entities)
        else:
            # Run individual strategies
            for strat_name in strategies:
                if strat_name in strategy_map:
                    factory = strategy_map[strat_name]
                    expander = factory() if callable(factory) else factory
                    try:
                        result = await expander.expand(query)
                        all_variations.extend(result.variations)
                        all_synonyms.update(result.synonyms)
                        all_keywords.extend(result.keywords)
                        all_entities.extend(result.entities)
                    except _QUERY_STRATEGY_EXCEPTIONS:
                        logger.debug("Query expansion strategy failed")

        # Deduplicate
        seen = set()
        unique_variations = []
        for v in all_variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)

        unique_variations = unique_variations[:max_expansions]
        unique_keywords = list(dict.fromkeys(all_keywords))[:20]
        unique_entities = list(dict.fromkeys(all_entities))[:10]

        # Build combined query string for downstream use
        combined_parts = [query] + unique_variations[:2]
        combined = " ".join(combined_parts)

        return {
            "original": query,
            "variations": unique_variations,
            "synonyms": all_synonyms,
            "keywords": unique_keywords,
            "entities": unique_entities,
            "combined": combined,
            "strategies_used": strategies,
        }

    except _QUERY_ADAPTER_EXCEPTIONS:
        logger.exception("Query expand adapter error")
        return {"error": "query_expand_error"}


@registry.register(
    "hyde_generate",
    category="rag",
    description="Generate hypothetical documents (HyDE)",
    parallelizable=True,
    tags=["rag", "query"],
    config_model=HyDEGenerateConfig,
)
async def run_hyde_generate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate a Hypothetical Document Embedding (HyDE) for improved similarity search.

    Config:
      - query: str (required, templated) - The search query
      - provider: str - LLM provider
      - model: str - Model to use
      - num_hypothetical: int = 1 - Number of hypothetical documents
      - document_type: Literal["answer", "passage", "article"] = "passage"
    Output:
      - {"query": str, "hypothetical_documents": [str], "document_type": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = apply_template_to_string(query, context) or query
    query = str(query).strip()

    if not query:
        return {"error": "missing_query", "query": "", "hypothetical_documents": []}

    provider = config.get("provider")
    model = config.get("model")
    num_hypothetical = int(config.get("num_hypothetical", 1))
    document_type = str(config.get("document_type", "passage")).lower()

    type_prompts = {
        "answer": "Write a direct, factual answer to the question as if you were an expert.",
        "passage": "Write a passage from a document that would contain the answer to this query.",
        "article": "Write an excerpt from an informative article that addresses this topic.",
    }

    system_prompt = f"""You are generating hypothetical documents for semantic search.
{type_prompts.get(document_type, type_prompts['passage'])}

Generate {num_hypothetical} hypothetical document(s). If multiple, separate with ---."""

    user_prompt = f"Query: {query}"

    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        messages = [{"role": "user", "content": user_prompt}]
        response = await perform_chat_api_call_async(
            messages=messages,
            api_provider=provider,
            model=model,
            system_message=system_prompt,
            max_tokens=1000,
            temperature=0.8,
        )

        text = extract_openai_content(response) or ""
        if num_hypothetical > 1:
            docs = [d.strip() for d in text.split("---") if d.strip()][:num_hypothetical]
        else:
            docs = [text.strip()] if text.strip() else []

        return {
            "query": query,
            "hypothetical_documents": docs,
            "document_type": document_type,
        }

    except _QUERY_ADAPTER_EXCEPTIONS:
        logger.exception("HyDE generate adapter error")
        return {"error": "hyde_error", "query": query, "hypothetical_documents": []}


@registry.register(
    "semantic_cache_check",
    category="rag",
    description="Check semantic cache",
    parallelizable=True,
    tags=["rag", "cache"],
    config_model=SemanticCacheCheckConfig,
)
async def run_semantic_cache_check_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Check semantic cache for similar queries before running expensive searches.

    Config:
      - query: str (required, templated) - Query to check
      - cache_collection: str = "semantic_cache" - ChromaDB collection for cache
      - similarity_threshold: float = 0.9 - Minimum similarity for cache hit
      - max_age_seconds: int = 3600 - Maximum age of cached results
    Output:
      - {"cache_hit": bool, "cached_query": str, "cached_result": dict,
         "similarity": float, "query": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    query = config.get("query") or ""
    if isinstance(query, str):
        query = apply_template_to_string(query, context) or query
    query = str(query).strip()

    if not query:
        return {"cache_hit": False, "query": "", "error": "missing_query"}

    cache_collection = config.get("cache_collection", "semantic_cache")
    similarity_threshold = float(config.get("similarity_threshold", 0.9))
    max_age_seconds = int(config.get("max_age_seconds", 3600))

    try:
        import time

        manager, collection = _get_semantic_cache_collection(cache_collection)
        if not collection:
            return {"cache_hit": False, "query": query, "error": "chroma_unavailable"}

        query_embedding = _build_semantic_cache_query_embedding(query, manager)

        # Search for similar queries
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances", "documents"],
        )

        if results and results.get("distances") and results["distances"][0]:
            distance = results["distances"][0][0]
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - distance

            if similarity >= similarity_threshold:
                metadata = results.get("metadatas", [[]])[0]
                if metadata:
                    meta = metadata[0] if isinstance(metadata, list) and metadata else metadata
                    cached_at = meta.get("cached_at", 0)
                    if time.time() - cached_at <= max_age_seconds:
                        cached_query = results.get("documents", [[]])[0]
                        if isinstance(cached_query, list) and cached_query:
                            cached_query = cached_query[0]

                        cached_result = meta.get("result")
                        if isinstance(cached_result, str):
                            with contextlib.suppress(_QUERY_JSON_PARSE_EXCEPTIONS):
                                cached_result = json.loads(cached_result)

                        return {
                            "cache_hit": True,
                            "query": query,
                            "cached_query": cached_query,
                            "cached_result": cached_result,
                            "similarity": similarity,
                        }

        return {"cache_hit": False, "query": query}

    except _QUERY_ADAPTER_EXCEPTIONS:
        logger.exception("Semantic cache check error")
        return {"cache_hit": False, "query": query, "error": "semantic_cache_error"}


@registry.register(
    "search_aggregate",
    category="rag",
    description="Aggregate search results",
    parallelizable=False,
    tags=["rag", "search"],
    config_model=SearchAggregateConfig,
)
async def run_search_aggregate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Aggregate and deduplicate results from multiple search steps.

    Config:
      - results: list[dict] - List of search results to aggregate
      - dedup_field: str = "id" - Field to use for deduplication
      - sort_by: str = "score" - Field to sort by
      - sort_order: Literal["asc", "desc"] = "desc"
      - limit: int = 20 - Maximum results to return
      - merge_scores: Literal["max", "sum", "avg"] = "max" - How to merge scores
    Output:
      - {"documents": [dict], "total_before_dedup": int,
         "total_after_dedup": int, "sources": [str]}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # Get results from config or from previous step
    results = config.get("results")
    if not results:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            results = prev.get("results")
            if not results and prev.get("documents"):
                results = [{"documents": prev.get("documents"), "source": prev.get("source", "prev")}]

    if not isinstance(results, list):
        results = [results] if results else []

    dedup_field = config.get("dedup_field", "id")
    sort_by = config.get("sort_by", "score")
    sort_order = config.get("sort_order", "desc")
    limit = int(config.get("limit", 20))
    merge_scores = config.get("merge_scores", "max")

    # Flatten all documents
    all_docs = []
    sources = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            docs = result.get("documents") or result.get("results") or []
            source = result.get("source", f"source_{i}")
        elif isinstance(result, list):
            docs = result
            source = f"source_{i}"
        else:
            continue

        sources.append(source)
        for doc in docs:
            if isinstance(doc, dict):
                doc_copy = dict(doc)
                doc_copy["_source"] = source
                all_docs.append(doc_copy)

    total_before = len(all_docs)

    # Deduplicate
    seen = {}
    for doc in all_docs:
        key = doc.get(dedup_field)
        if key is None:
            # Generate key from content
            key = hash(str(doc.get("content", doc.get("text", str(doc)))))

        if key in seen:
            # Merge scores
            existing = seen[key]
            existing_score = existing.get(sort_by, 0)
            new_score = doc.get(sort_by, 0)

            if merge_scores == "sum":
                existing[sort_by] = existing_score + new_score
            elif merge_scores == "avg":
                existing["_count"] = existing.get("_count", 1) + 1
                existing[sort_by] = (existing_score * (existing["_count"] - 1) + new_score) / existing["_count"]
            else:  # max
                existing[sort_by] = max(existing_score, new_score)
        else:
            seen[key] = doc

    # Sort
    deduped = list(seen.values())
    reverse = sort_order == "desc"
    deduped.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)

    # Limit
    final_docs = deduped[:limit]

    # Clean up internal fields
    for doc in final_docs:
        doc.pop("_count", None)

    return {
        "documents": final_docs,
        "total_before_dedup": total_before,
        "total_after_dedup": len(deduped),
        "sources": sources,
    }
