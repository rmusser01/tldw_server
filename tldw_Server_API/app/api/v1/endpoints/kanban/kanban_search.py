# app/api/v1/endpoints/kanban_search.py
"""
Kanban Search API endpoints.

Provides search functionality for Kanban cards including:
- FTS5 full-text search
- Optional vector/hybrid search (when ChromaDB is available)

Hybrid Search Scoring
---------------------
When hybrid search is enabled (requires ChromaDB), results are scored using a
weighted combination of full-text search (FTS) and vector similarity scores.

The scoring formula for cards found by both FTS and vector search is:
    hybrid_score = (FTS_WEIGHT * fts_score) + (VECTOR_WEIGHT * vector_score)

For cards found only by vector search (semantic match without keyword match),
the score is reduced:
    score = VECTOR_ONLY_WEIGHT * vector_score

Default weights prioritize keyword relevance (FTS) while incorporating
semantic similarity (vector):
- FTS_WEIGHT: 0.6 (60%) - Keyword matches are primary
- VECTOR_WEIGHT: 0.4 (40%) - Semantic similarity supplements FTS
- VECTOR_ONLY_WEIGHT: 0.3 (30%) - Pure semantic matches ranked lower

These weights can be customized via environment variables:
- KANBAN_SEARCH_FTS_WEIGHT
- KANBAN_SEARCH_VECTOR_WEIGHT
- KANBAN_SEARCH_VECTOR_ONLY_WEIGHT
"""
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
    handle_kanban_db_error,
    kanban_rate_limit,
)
from tldw_Server_API.app.api.v1.endpoints.kanban._kanban_utils import resolve_limit_offset
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.kanban_schemas import (
    PaginationInfo,
    SearchRequest,
    SearchResponse,
    SearchResultCard,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    KanbanDB,
    KanbanDBError,
)
from tldw_Server_API.app.core.DB_Management.kanban_vector_search import (
    KanbanVectorSearch,
    is_vector_search_available,
)

# --- Search Scoring Configuration ---
# Weights for hybrid search scoring (configurable via environment variables)
#
# FTS_WEIGHT: Weight for full-text search score in hybrid mode.
#   Higher values prioritize keyword/phrase matches.
#   Default: 0.6 (60% weight on keyword matching)
#
# VECTOR_WEIGHT: Weight for vector similarity score in hybrid mode.
#   Higher values prioritize semantic similarity.
#   Default: 0.4 (40% weight on semantic matching)
#
# VECTOR_ONLY_WEIGHT: Multiplier for cards found only by vector search.
#   These cards matched semantically but had no keyword match.
#   Lower values rank pure semantic matches below keyword matches.
#   Default: 0.3 (semantic-only results get 30% of their vector score)
#
# Note: FTS_WEIGHT + VECTOR_WEIGHT should sum to 1.0 for normalized scoring.

def _get_float_env(key: str, default: float) -> float:
    """Get a float value from environment variable with fallback default."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {value}, using default {default}")
        return default


FTS_WEIGHT = _get_float_env("KANBAN_SEARCH_FTS_WEIGHT", 0.6)
VECTOR_WEIGHT = _get_float_env("KANBAN_SEARCH_VECTOR_WEIGHT", 0.4)
VECTOR_ONLY_WEIGHT = _get_float_env("KANBAN_SEARCH_VECTOR_ONLY_WEIGHT", 0.3)


router = APIRouter(tags=["Kanban Search"])


# --- Helper for Exception Handling ---
def _handle_error(e: Exception) -> HTTPException:
    """Convert exceptions to appropriate HTTP responses."""
    return handle_kanban_db_error(e)


def _card_to_search_result(card: dict) -> SearchResultCard:
    """Convert a card dict to SearchResultCard schema."""
    return SearchResultCard(
        id=card["id"],
        uuid=card["uuid"],
        board_id=card["board_id"],
        board_name=card.get("board_name", ""),
        list_id=card["list_id"],
        list_name=card.get("list_name", ""),
        title=card["title"],
        description=card.get("description"),
        priority=card.get("priority"),
        due_date=card.get("due_date"),
        labels=card.get("labels", []),
        created_at=card["created_at"],
        updated_at=card["updated_at"],
        relevance_score=card.get("relevance_score"),
    )


def _execute_search(
    db: KanbanDB,
    query: str,
    board_id: Optional[int],
    label_ids: Optional[list[int]],
    priority: Optional[str],
    include_archived: bool,
    search_mode: str,
    limit: int,
    offset: int,
) -> SearchResponse:
    """
    Execute search and return SearchResponse.

    Centralizes the search execution logic shared between GET and POST endpoints.

    Args:
        db: KanbanDB instance
        query: Search query string
        board_id: Optional board ID filter
        label_ids: Optional list of label IDs (cards must have ALL)
        priority: Optional priority filter
        include_archived: Whether to include archived cards
        search_mode: Search mode (fts, vector, hybrid)
        limit: Maximum results
        offset: Offset for pagination

    Returns:
        SearchResponse with results and pagination info
    """
    # Get vector search if available (for vector/hybrid modes)
    vector_search: Optional[KanbanVectorSearch] = None
    if search_mode in ["vector", "hybrid"]:
        vector_search = db.get_vector_search()

    # Perform search based on mode
    if search_mode == "fts" or search_mode not in ["fts", "vector", "hybrid"]:
        cards, total = _perform_fts_search(
            db, query, board_id, label_ids, priority, include_archived, limit, offset
        )
        actual_mode = "fts"
    elif search_mode == "vector":
        cards, total, actual_mode = _perform_vector_search(
            db, vector_search, query, board_id, label_ids, priority,
            include_archived, limit, offset
        )
    else:  # hybrid
        cards, total, actual_mode = _perform_hybrid_search(
            db, vector_search, query, board_id, label_ids, priority,
            include_archived, limit, offset
        )

    # Convert to response schema
    results = [_card_to_search_result(card) for card in cards]

    return SearchResponse(
        query=query,
        search_mode=actual_mode,
        results=results,
        pagination=PaginationInfo(
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + len(results)) < total
        )
    )


def _perform_fts_search(
    db: KanbanDB,
    query: str,
    board_id: Optional[int],
    label_ids: Optional[list[int]],
    priority: Optional[str],
    include_archived: bool,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    """Perform FTS5 search."""
    return db.search_cards(
        query=query,
        board_id=board_id,
        label_ids=label_ids,
        priority=priority,
        include_archived=include_archived,
        limit=limit,
        offset=offset,
    )


def _perform_vector_search(
    db: KanbanDB,
    vector_search: Optional[KanbanVectorSearch],
    query: str,
    board_id: Optional[int],
    label_ids: Optional[list[int]],
    priority: Optional[str],
    include_archived: bool,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int, str]:
    """
    Perform vector search with fallback to FTS.

    Returns: (cards, total, actual_mode)
    """
    if vector_search is None or not vector_search.available:
        logger.debug("Vector search unavailable, falling back to FTS")
        cards, total = _perform_fts_search(
            db, query, board_id, label_ids, priority, include_archived, limit, offset
        )
        return cards, total, "fts"

    try:
        # Get vector search results
        vector_results = vector_search.search(
            query=query,
            board_id=board_id,
            priority=priority,
            limit=limit + offset,  # Get extra for pagination
        )

        if not vector_results:
            # No vector results, fall back to FTS
            logger.debug("No vector results, falling back to FTS")
            cards, total = _perform_fts_search(
                db, query, board_id, label_ids, priority, include_archived, limit, offset
            )
            return cards, total, "fts"

        # Get card IDs from vector results
        card_ids = [r["card_id"] for r in vector_results if r.get("card_id")]
        relevance_map = {r["card_id"]: r["relevance_score"] for r in vector_results}

        # Batch fetch cards (more efficient than N+1 queries)
        fetched_cards = db.get_cards_by_ids(
            card_ids=card_ids,
            include_deleted=False,
            include_archived=include_archived
        )

        # Apply additional filters and add relevance scores
        cards = []
        for card in fetched_cards:
            # Apply label filter (vector search doesn't support this)
            if label_ids:
                card_label_ids = {l["id"] for l in card.get("labels", [])}
                if not set(label_ids).issubset(card_label_ids):
                    continue

            # Add relevance score
            card["relevance_score"] = relevance_map.get(card["id"], 0.0)
            cards.append(card)

        # Apply pagination
        total = len(cards)
        cards = cards[offset:offset + limit]

        return cards, total, "vector"

    except Exception as e:
        logger.warning("Vector search failed, falling back to FTS")
        cards, total = _perform_fts_search(
            db, query, board_id, label_ids, priority, include_archived, limit, offset
        )
        return cards, total, "fts"


def _perform_hybrid_search(
    db: KanbanDB,
    vector_search: Optional[KanbanVectorSearch],
    query: str,
    board_id: Optional[int],
    label_ids: Optional[list[int]],
    priority: Optional[str],
    include_archived: bool,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int, str]:
    """
    Perform hybrid search combining FTS and vector results.

    Returns: (cards, total, actual_mode)
    """
    # Always get FTS results as baseline
    fts_cards, fts_total = _perform_fts_search(
        db, query, board_id, label_ids, priority, include_archived, limit * 2, 0
    )

    # If vector search unavailable, just return FTS
    if vector_search is None or not vector_search.available:
        logger.debug("Vector search unavailable for hybrid, using FTS only")
        return fts_cards[offset:offset + limit], fts_total, "fts"

    try:
        # Get vector results
        vector_results = vector_search.search(
            query=query,
            board_id=board_id,
            priority=priority,
            limit=limit * 2,
        )

        if not vector_results:
            # No vector results, just use FTS
            return fts_cards[offset:offset + limit], fts_total, "fts"

        # Build a map of card_id -> vector relevance score
        vector_scores: dict[int, float] = {
            r["card_id"]: r["relevance_score"]
            for r in vector_results
            if r.get("card_id")
        }

        # Combine results: union of FTS and vector
        seen_ids: set[int] = set()
        combined_cards: list[dict[str, Any]] = []

        # Score FTS results (give them a base score)
        for i, card in enumerate(fts_cards):
            card_id = card["id"]
            if card_id in seen_ids:
                continue
            seen_ids.add(card_id)

            # FTS score: 1.0 - (position / total) for ranking
            fts_score = 1.0 - (i / max(len(fts_cards), 1))
            # Combine with vector score if available
            vector_score = vector_scores.get(card_id, 0.0)

            # Hybrid score: weighted average (60% FTS, 40% vector for keyword relevance)
            hybrid_score = (FTS_WEIGHT * fts_score) + (VECTOR_WEIGHT * vector_score)
            card["relevance_score"] = hybrid_score

            combined_cards.append(card)

        # Add any vector-only results (cards that matched semantically but not keyword)
        # Collect IDs that need to be fetched (not already in FTS results)
        vector_only_ids = [cid for cid in vector_scores if cid not in seen_ids]

        if vector_only_ids:
            # Batch fetch all vector-only cards in a single query
            try:
                vector_only_cards = db.get_cards_by_ids(
                    card_ids=vector_only_ids,
                    include_deleted=False,
                    include_archived=include_archived
                )

                for card in vector_only_cards:
                    card_id = card["id"]
                    if card_id in seen_ids:
                        continue

                    # Apply label filter
                    if label_ids:
                        card_label_ids = {l["id"] for l in card.get("labels", [])}
                        if not set(label_ids).issubset(card_label_ids):
                            continue

                    # Vector-only results get lower base score (semantic but not keyword match)
                    vector_score = vector_scores.get(card_id, 0.0)
                    card["relevance_score"] = VECTOR_ONLY_WEIGHT * vector_score

                    combined_cards.append(card)
                    seen_ids.add(card_id)

            except Exception as e:
                logger.warning("Failed to fetch vector-only cards in hybrid search")

        # Sort by combined relevance
        combined_cards.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)

        total = len(combined_cards)
        paginated = combined_cards[offset:offset + limit]

        return paginated, total, "hybrid"

    except Exception as e:
        logger.warning("Hybrid search failed, using FTS only")
        return fts_cards[offset:offset + limit], fts_total, "fts"


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search cards",
    description="Search cards using FTS5 full-text search.",
    dependencies=[Depends(kanban_rate_limit("kanban.search"))]
)
async def search_cards_get(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    board_id: Optional[int] = Query(None, description="Filter by board ID"),
    label_ids: Optional[str] = Query(None, description="Comma-separated label IDs"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    include_archived: bool = Query(False, description="Include archived cards"),
    search_mode: str = Query("fts", description="Search mode: fts, vector, or hybrid"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    page: Optional[int] = Query(None, ge=1, description="Legacy page number (1-indexed)"),
    per_page: Optional[int] = Query(None, ge=1, le=100, description="Legacy page size"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> SearchResponse:
    """
    Search cards using keyword query.

    - **q**: Search query (required)
    - **board_id**: Optional board ID to filter results
    - **label_ids**: Comma-separated label IDs (cards must have ALL)
    - **priority**: Filter by priority (low, medium, high, urgent)
    - **include_archived**: Include archived cards in results
    - **search_mode**: fts (default), vector, or hybrid
    - **limit**: Maximum results (max 100)
    - **offset**: Results to skip

    Note: Vector and hybrid search modes require ChromaDB integration.
    If unavailable, falls back to FTS.
    """
    try:
        # Parse label_ids from comma-separated string
        parsed_label_ids: Optional[list[int]] = None
        if label_ids:
            parsed_label_ids = [int(lid.strip()) for lid in label_ids.split(",")]

        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)

        return _execute_search(
            db=db,
            query=q,
            board_id=board_id,
            label_ids=parsed_label_ids,
            priority=priority,
            include_archived=include_archived,
            search_mode=search_mode,
            limit=limit,
            offset=offset,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid label_ids format: {str(e)}"
        ) from e
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to search cards") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search cards (POST)",
    description="Search cards using FTS5 full-text search with request body.",
    dependencies=[Depends(kanban_rate_limit("kanban.search"))]
)
async def search_cards_post(
    request: SearchRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> SearchResponse:
    """
    Search cards using keyword query (POST version with request body).

    This endpoint accepts the same parameters as the GET version but in the request body.
    Useful for complex queries or when query parameters are too long.
    """
    try:
        return _execute_search(
            db=db,
            query=request.query,
            board_id=request.board_id,
            label_ids=request.label_ids,
            priority=request.priority,
            include_archived=request.include_archived,
            search_mode=request.search_mode,
            limit=request.limit,
            offset=request.offset,
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to search cards") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/search/status",
    summary="Get search status",
    description="Get the status of search capabilities and scoring configuration.",
    dependencies=[Depends(kanban_rate_limit("kanban.search.status"))]
)
async def search_status(
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> dict[str, Any]:
    """
    Get the status of search capabilities.

    Returns information about which search modes are available and
    the current hybrid search scoring weights.
    """
    vector_search = db.get_vector_search()
    vector_enabled = bool(vector_search and vector_search.available)
    vector_backend = is_vector_search_available()
    return {
        "fts_available": True,  # Always available (SQLite FTS5)
        "vector_available": vector_enabled,
        "vector_backend_available": vector_backend,
        "hybrid_available": vector_enabled,
        "default_mode": "fts",
        "supported_modes": ["fts", "vector", "hybrid"],
        "scoring_weights": {
            "fts_weight": FTS_WEIGHT,
            "vector_weight": VECTOR_WEIGHT,
            "vector_only_weight": VECTOR_ONLY_WEIGHT,
        },
    }
