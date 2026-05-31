# app/api/v1/endpoints/kanban_links.py
"""
Kanban Card Links API endpoints (Phase 5: Content Integration).

Provides endpoints to link Kanban cards to media items and notes,
enabling bidirectional lookups between task management and content.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
    kanban_rate_limit,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.kanban_schemas import (
    BulkCardLinksAddResponse,
    BulkCardLinksRemoveResponse,
    BulkCardLinksRequest,
    CardLinkCountsResponse,
    CardLinkCreate,
    CardLinkResponse,
    CardLinksListResponse,
    DetailResponse,
    LinkedCardResponse,
    LinkedCardsListResponse,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)

router = APIRouter(tags=["kanban"])

ALLOWED_LINKED_TYPES = ("media", "note")


def _validate_linked_type_or_400(linked_type: Optional[str]) -> Optional[str]:
    """
    Validate linked_type for endpoints that accept 'media' or 'note'.

    Returns the value when valid, raises HTTP 400 for invalid types.
    """
    if linked_type is None:
        return None
    if linked_type not in ALLOWED_LINKED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid linked_type: {linked_type}. Must be 'media' or 'note'.",
        )
    return linked_type


# =============================================================================
# Card Link CRUD Endpoints
# =============================================================================

@router.post(
    "/cards/{card_id}/links",
    response_model=CardLinkResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a link to a card",
    description="Link a card to a media item or note.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.create"))]
)
async def add_card_link(
    card_id: int,
    link_in: CardLinkCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardLinkResponse:
    """
    Add a link from a card to a media item or note.

    - **linked_type**: Type of content to link ('media' or 'note')
    - **linked_id**: ID of the content to link
    """
    try:
        link = db.add_card_link(
            card_id=card_id,
            linked_type=link_in.linked_type,
            linked_id=link_in.linked_id
        )
        logger.info(f"Added link from card {card_id} to {link_in.linked_type}:{link_in.linked_id}")
        return CardLinkResponse(**link)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to add card link") from e
@router.get(
    "/cards/{card_id}/links",
    response_model=CardLinksListResponse,
    summary="Get all links for a card",
    description="Get all media and note links for a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.list"))]
)
async def get_card_links(
    card_id: int,
    linked_type: Optional[str] = Query(None, description="Filter by type ('media' or 'note')"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardLinksListResponse:
    """
    Get all links for a card.

    Optionally filter by linked_type.
    """
    try:
        linked_type = _validate_linked_type_or_400(linked_type)
        links = db.get_card_links(card_id=card_id, linked_type=linked_type)
        return CardLinksListResponse(links=[CardLinkResponse(**link) for link in links])
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch card links") from e
@router.get(
    "/cards/{card_id}/links/counts",
    response_model=CardLinkCountsResponse,
    summary="Get link counts for a card",
    description="Get counts of linked content by type."
)
async def get_card_link_counts(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardLinkCountsResponse:
    """
    Get counts of linked content by type for a card.

    Returns {"media": N, "note": M}.
    """
    try:
        counts = db.get_linked_content_counts(card_id=card_id)
        return CardLinkCountsResponse(**counts)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch card link counts") from e
@router.delete(
    "/cards/{card_id}/links/{linked_type}/{linked_id}",
    response_model=DetailResponse,
    summary="Remove a link from a card",
    description="Remove a specific link from a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.delete"))]
)
async def remove_card_link(
    card_id: int,
    linked_type: str,
    linked_id: str,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """
    Remove a link from a card.

    - **linked_type**: Type of linked content ('media' or 'note')
    - **linked_id**: ID of the linked content
    """
    linked_type = _validate_linked_type_or_400(linked_type)
    try:
        removed = db.remove_card_link(
            card_id=card_id,
            linked_type=linked_type,
            linked_id=linked_id
        )
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Link not found for card {card_id} to {linked_type}:{linked_id}"
            )
        logger.info(f"Removed link from card {card_id} to {linked_type}:{linked_id}")
        return DetailResponse(detail="Link removed successfully")
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to remove card link") from e
@router.delete(
    "/cards/{card_id}/links/{link_id}",
    response_model=DetailResponse,
    summary="Remove a link from a card by ID",
    description="Remove a card link by its ID.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.delete"))]
)
async def remove_card_link_by_id_for_card(
    card_id: int,
    link_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """Remove a card link by ID scoped to a card."""
    try:
        removed = db.remove_card_link_by_id_for_card(card_id=card_id, link_id=link_id)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Link {link_id} not found for card {card_id}"
            )
        logger.info(f"Removed link {link_id} from card {card_id}")
        return DetailResponse(detail="Link removed successfully")
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to remove card link") from e
@router.delete(
    "/links/{link_id}",
    response_model=DetailResponse,
    summary="Remove a link by ID",
    description="Remove a card link by its ID.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.delete"))]
)
async def remove_card_link_by_id(
    link_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """Remove a card link by its ID."""
    try:
        removed = db.remove_card_link_by_id(link_id=link_id)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Link {link_id} not found"
            )
        logger.info(f"Removed link {link_id}")
        return DetailResponse(detail="Link removed successfully")
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to remove card link") from e
# =============================================================================
# Bulk Link Operations
# =============================================================================

@router.post(
    "/cards/{card_id}/links/bulk-add",
    response_model=BulkCardLinksAddResponse,
    summary="Bulk add links to a card",
    description="Add multiple links to a card at once.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.bulk"))]
)
async def bulk_add_card_links(
    card_id: int,
    request: BulkCardLinksRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkCardLinksAddResponse:
    """
    Add multiple links to a card at once.

    Duplicate links are skipped (counted in skipped_count).
    """
    try:
        # Convert Pydantic models to dicts
        links = [{"linked_type": link.linked_type, "linked_id": link.linked_id} for link in request.links]
        result = db.bulk_add_card_links(card_id=card_id, links=links)
        logger.info(f"Bulk added {result['added_count']} links to card {card_id}")
        return BulkCardLinksAddResponse(**result)
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk add card links") from e
@router.post(
    "/cards/{card_id}/links/bulk-remove",
    response_model=BulkCardLinksRemoveResponse,
    summary="Bulk remove links from a card",
    description="Remove multiple links from a card at once.",
    dependencies=[Depends(kanban_rate_limit("kanban.links.bulk"))]
)
async def bulk_remove_card_links(
    card_id: int,
    request: BulkCardLinksRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkCardLinksRemoveResponse:
    """
    Remove multiple links from a card at once.

    Non-existent links are silently ignored.
    """
    try:
        # Convert Pydantic models to dicts
        links = [{"linked_type": link.linked_type, "linked_id": link.linked_id} for link in request.links]
        result = db.bulk_remove_card_links(card_id=card_id, links=links)
        logger.info(f"Bulk removed {result['removed_count']} links from card {card_id}")
        return BulkCardLinksRemoveResponse(**result)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk remove card links") from e
# =============================================================================
# Bidirectional Lookup Endpoints
# =============================================================================

@router.get(
    "/linked/{linked_type}/{linked_id}/cards",
    response_model=LinkedCardsListResponse,
    summary="Get cards linked to content",
    description="Find all cards that link to a specific media item or note (bidirectional lookup).",
    dependencies=[Depends(kanban_rate_limit("kanban.links.lookup"))]
)
async def get_cards_by_linked_content(
    linked_type: str,
    linked_id: str,
    include_archived: bool = Query(False, description="Include archived cards"),
    include_deleted: bool = Query(False, description="Include soft-deleted cards"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> LinkedCardsListResponse:
    """
    Find all cards that link to a specific media item or note.

    This is the bidirectional lookup endpoint - given a media item or note ID,
    find all Kanban cards that reference it.

    - **linked_type**: Type of content ('media' or 'note')
    - **linked_id**: ID of the content
    - **include_archived**: Include archived cards in results
    - **include_deleted**: Include soft-deleted cards in results
    """
    linked_type = _validate_linked_type_or_400(linked_type)

    try:
        cards = db.get_cards_by_linked_content(
            linked_type=linked_type,
            linked_id=linked_id,
            include_archived=include_archived,
            include_deleted=include_deleted
        )
        return LinkedCardsListResponse(
            linked_type=linked_type,
            linked_id=linked_id,
            cards=[LinkedCardResponse(**card) for card in cards]
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch linked cards") from e
