# app/api/v1/endpoints/kanban_cards.py
"""
Kanban Card API endpoints.

Provides CRUD operations for Kanban cards including:
- Create, read, update cards
- Archive/unarchive cards
- Soft delete and restore cards
- Move and copy cards between lists
- Reorder cards within a list
- Search cards
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
    handle_kanban_db_error,
    kanban_rate_limit,
)
from tldw_Server_API.app.api.v1.endpoints.kanban._kanban_utils import (
    resolve_limit_offset,
    to_db_timestamp,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.kanban_schemas import (
    ActivitiesListResponse,
    ActivityResponse,
    BulkArchiveCardsRequest,
    BulkArchiveCardsResponse,
    BulkDeleteCardsRequest,
    BulkDeleteCardsResponse,
    BulkLabelCardsRequest,
    BulkLabelCardsResponse,
    BulkMoveCardsRequest,
    BulkMoveCardsResponse,
    BulkUnarchiveCardsResponse,
    CardCopyRequest,
    CardCopyWithChecklistsRequest,
    CardCreate,
    CardMoveRequest,
    CardResponse,
    CardSearchRequest,
    CardSearchResponse,
    CardsListResponse,
    CardUpdate,
    CardWithDetailsResponse,
    DetailResponse,
    FilteredCardsResponse,
    PaginationInfo,
    ReorderRequest,
    ReorderResponse,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)

router = APIRouter(tags=["Kanban Cards"])


# --- Helper for Exception Handling ---
def _handle_error(e: Exception) -> HTTPException:
    """Convert exceptions to appropriate HTTP responses."""
    return handle_kanban_db_error(e)


# =============================================================================
# Card CRUD Endpoints (nested under /lists/{list_id}/cards)
# =============================================================================

@router.post(
    "/lists/{list_id}/cards",
    response_model=CardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new card",
    description="Create a new card in a list.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.create"))]
)
async def create_card(
    list_id: int,
    card_in: CardCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """
    Create a new card in a list.

    - **title**: Card title (required, 1-500 characters)
    - **description**: Optional card description
    - **client_id**: Client-generated unique ID for idempotency
    - **position**: Optional position (0-indexed, defaults to end)
    - **due_date**: Optional due date
    - **start_date**: Optional start date
    - **priority**: Optional priority (low, medium, high, urgent)
    - **metadata**: Optional JSON metadata
    """
    try:
        # Convert datetime to string if provided
        due_date_str = card_in.due_date.isoformat() if card_in.due_date else None
        start_date_str = card_in.start_date.isoformat() if card_in.start_date else None

        card = db.create_card(
            list_id=list_id,
            title=card_in.title,
            client_id=card_in.client_id,
            description=card_in.description,
            position=card_in.position,
            due_date=due_date_str,
            start_date=start_date_str,
            priority=card_in.priority,
            metadata=card_in.metadata
        )
        if card_in.label_ids:
            db.bulk_label_cards(
                card_ids=[card["id"]],
                add_label_ids=list(dict.fromkeys(card_in.label_ids)),
            )
        logger.info(f"Created card {card['id']} in list {list_id}")
        return CardResponse(**card)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to create card") from e
@router.get(
    "/lists/{list_id}/cards",
    response_model=CardsListResponse,
    summary="Get all cards in a list",
    description="Get all cards for a list, ordered by position.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.list"))]
)
async def get_cards(
    list_id: int,
    include_archived: bool = Query(False, description="Include archived cards"),
    include_deleted: bool = Query(False, description="Include soft-deleted cards"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardsListResponse:
    """
    Get all cards in a list.

    Cards are returned ordered by position (ascending).
    """
    try:
        cards = db.list_cards(
            list_id=list_id,
            include_archived=include_archived,
            include_deleted=include_deleted
        )
        return CardsListResponse(cards=[CardResponse(**c) for c in cards])
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch cards") from e


@router.post(
    "/lists/{list_id}/cards/reorder",
    response_model=ReorderResponse,
    summary="Reorder cards in a list",
    description="Set the order of cards in a list.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.reorder"))]
)
async def reorder_cards(
    list_id: int,
    reorder_in: ReorderRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ReorderResponse:
    """
    Reorder cards in a list.

    Provide the card IDs in the desired order. All active (non-deleted) cards
    in the list must be included.
    """
    try:
        ids = reorder_in.ids or []
        db.reorder_cards(list_id=list_id, card_ids=ids)
        logger.info(f"Reordered {len(ids)} cards in list {list_id}")
        return ReorderResponse(success=True, message="Cards reordered successfully")
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to reorder cards") from e
# =============================================================================
# Individual Card Endpoints (at /cards/{card_id})
# =============================================================================

@router.get(
    "/cards/{card_id}",
    response_model=CardWithDetailsResponse,
    summary="Get a card",
    description="Get a card by ID.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.get"))]
)
async def get_card(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardWithDetailsResponse:
    """Get a card by ID."""
    try:
        card = db.get_card_with_details(card_id)
        if not card:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Card {card_id} not found"
            )
        return CardWithDetailsResponse(**card)
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch card") from e
@router.patch(
    "/cards/{card_id}",
    response_model=CardResponse,
    summary="Update a card",
    description="Update card properties.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.update"))]
)
async def update_card(
    card_id: int,
    card_in: CardUpdate,
    x_expected_version: Optional[int] = Header(None, description="Expected version for optimistic locking"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """
    Update a card.

    All fields are optional. Only provided fields will be updated.
    Use the X-Expected-Version header for optimistic locking.
    """
    try:
        # Convert datetime to string if provided
        due_date_str = card_in.due_date.isoformat() if card_in.due_date else None
        start_date_str = card_in.start_date.isoformat() if card_in.start_date else None

        card = db.update_card(
            card_id=card_id,
            title=card_in.title,
            description=card_in.description,
            due_date=due_date_str,
            due_complete=card_in.due_complete,
            start_date=start_date_str,
            priority=card_in.priority,
            metadata=card_in.metadata,
            expected_version=x_expected_version
        )
        logger.info(f"Updated card {card_id}")
        return CardResponse(**card)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to update card") from e
# =============================================================================
# Move and Copy Operations
# =============================================================================

@router.post(
    "/cards/{card_id}/move",
    response_model=CardResponse,
    summary="Move a card to another list",
    description="Move a card to a different list within the same board.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.move"))]
)
async def move_card(
    card_id: int,
    move_in: CardMoveRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """
    Move a card to a different list.

    The target list must be in the same board as the card.
    """
    try:
        card = db.move_card(
            card_id=card_id,
            target_list_id=move_in.target_list_id,
            position=move_in.position
        )
        logger.info(f"Moved card {card_id} to list {move_in.target_list_id}")
        return CardResponse(**card)
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to move card") from e
@router.post(
    "/cards/{card_id}/copy",
    response_model=CardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Copy a card",
    description="Copy a card to a list (can be same or different list).",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.copy"))]
)
async def copy_card(
    card_id: int,
    copy_in: CardCopyRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """
    Copy a card to a list.

    Creates a copy of the card with its description, labels, metadata,
    and checklists.
    """
    try:
        card = db.copy_card_with_checklists(
            card_id=card_id,
            target_list_id=copy_in.target_list_id,
            new_client_id=copy_in.new_client_id,
            position=copy_in.position,
            new_title=copy_in.new_title,
            copy_checklists=True,
            copy_labels=True
        )
        logger.info(f"Copied card {card_id} to list {copy_in.target_list_id} as {card['id']}")
        return CardResponse(**card)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to copy card") from e
# =============================================================================
# Archive Operations
# =============================================================================

@router.post(
    "/cards/{card_id}/archive",
    response_model=CardResponse,
    summary="Archive a card",
    description="Archive a card. Archived cards are hidden from default listings but can be restored.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.update"))]
)
async def archive_card(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """Archive a card."""
    try:
        card = db.archive_card(card_id, archive=True)
        logger.info(f"Archived card {card_id}")
        return CardResponse(**card)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to archive card") from e
@router.post(
    "/cards/{card_id}/unarchive",
    response_model=CardResponse,
    summary="Unarchive a card",
    description="Restore an archived card to active status.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.update"))]
)
async def unarchive_card(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """Unarchive a card."""
    try:
        card = db.archive_card(card_id, archive=False)
        logger.info(f"Unarchived card {card_id}")
        return CardResponse(**card)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to unarchive card") from e
# =============================================================================
# Delete Operations
# =============================================================================

@router.delete(
    "/cards/{card_id}",
    response_model=DetailResponse,
    summary="Delete a card",
    description="Soft-delete a card. The card can be restored within the retention period.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.delete"))]
)
async def delete_card(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """Soft-delete a card."""
    try:
        success = db.delete_card(card_id, hard_delete=False)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Card {card_id} not found"
            )
        logger.info(f"Deleted card {card_id}")
        return DetailResponse(detail=f"Card {card_id} deleted successfully")
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete card") from e
@router.post(
    "/cards/{card_id}/restore",
    response_model=CardResponse,
    summary="Restore a deleted card",
    description="Restore a soft-deleted card.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.delete"))]
)
async def restore_card(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """Restore a soft-deleted card."""
    try:
        card = db.restore_card(card_id)
        logger.info(f"Restored card {card_id}")
        return CardResponse(**card)
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to restore card") from e
# =============================================================================
# Activity Endpoints
# =============================================================================

@router.get(
    "/cards/{card_id}/activities",
    response_model=ActivitiesListResponse,
    summary="Get card activities",
    description="Get activity log for a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.get"))]
)
async def get_card_activities(
    card_id: int,
    created_after: Optional[datetime] = Query(None, description="Filter by created_at >= timestamp"),
    created_before: Optional[datetime] = Query(None, description="Filter by created_at <= timestamp"),
    action_type: Optional[str] = Query(None, description="Filter by action_type"),
    entity_type: Optional[str] = Query(None, description="Filter by entity_type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum activities to return"),
    offset: int = Query(0, ge=0, description="Number of activities to skip"),
    page: Optional[int] = Query(None, ge=1, description="Legacy page number (1-indexed)"),
    per_page: Optional[int] = Query(None, ge=1, le=200, description="Legacy page size"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ActivitiesListResponse:
    """Get activity log for a card."""
    try:
        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)
        activities, total = db.get_card_activities(
            card_id=card_id,
            created_after=to_db_timestamp(created_after),
            created_before=to_db_timestamp(created_before),
            action_type=action_type,
            entity_type=entity_type,
            limit=limit,
            offset=offset,
        )
        return ActivitiesListResponse(
            activities=[ActivityResponse(**a) for a in activities],
            pagination=PaginationInfo(
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(activities)) < total
            )
        )
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch card activities") from e
# =============================================================================
# Search Operations
# =============================================================================

@router.post(
    "/cards/search",
    response_model=CardSearchResponse,
    summary="Search cards",
    description="Search cards using full-text search.",
    dependencies=[Depends(kanban_rate_limit("kanban.search"))]
)
async def search_cards(
    search_in: CardSearchRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardSearchResponse:
    """
    Search cards using FTS5 full-text search.

    Searches card titles and descriptions.
    """
    try:
        cards, total = db.search_cards(
            query=search_in.query,
            board_id=search_in.board_id,
            limit=search_in.limit,
            offset=search_in.offset
        )
        return CardSearchResponse(
            cards=[CardResponse(**c) for c in cards],
            pagination=PaginationInfo(
                total=total,
                limit=search_in.limit,
                offset=search_in.offset,
                has_more=(search_in.offset + len(cards)) < total
            )
        )
    except (InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to search cards") from e
@router.get(
    "/cards/search",
    response_model=CardSearchResponse,
    summary="Search cards (GET)",
    description="Search cards using full-text search via query parameters.",
    dependencies=[Depends(kanban_rate_limit("kanban.search"))]
)
async def search_cards_get(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    board_id: Optional[int] = Query(None, description="Filter by board ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    page: Optional[int] = Query(None, ge=1, description="Legacy page number (1-indexed)"),
    per_page: Optional[int] = Query(None, ge=1, le=200, description="Legacy page size"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardSearchResponse:
    """
    Search cards using FTS5 full-text search (GET variant).

    Searches card titles and descriptions.
    """
    try:
        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)
        cards, total = db.search_cards(
            query=q,
            board_id=board_id,
            limit=limit,
            offset=offset
        )
        return CardSearchResponse(
            cards=[CardResponse(**c) for c in cards],
            pagination=PaginationInfo(
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(cards)) < total
            )
        )
    except (InputError, KanbanDBError) as e:
        raise _handle_error(e) from e
# =============================================================================
# Bulk Operations Endpoints (Phase 3)
# =============================================================================

@router.post(
    "/cards/bulk-move",
    response_model=BulkMoveCardsResponse,
    summary="Bulk move cards",
    description="Move multiple cards to a target list.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.bulk"))]
)
async def bulk_move_cards(
    request: BulkMoveCardsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkMoveCardsResponse:
    """
    Move multiple cards to a target list.

    - **card_ids**: List of card IDs to move (must be in same board as target list)
    - **target_list_id**: Destination list ID
    - **position**: Optional starting position (cards placed sequentially from here)
    """
    try:
        result = db.bulk_move_cards(
            card_ids=request.card_ids,
            target_list_id=request.target_list_id,
            start_position=request.position
        )
        return BulkMoveCardsResponse(
            success=result["success"],
            moved_count=result["moved_count"],
            cards=[CardResponse(**c) for c in result["cards"]]
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk move cards") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/cards/bulk-archive",
    response_model=BulkArchiveCardsResponse,
    summary="Bulk archive cards",
    description="Archive multiple cards.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.bulk"))]
)
async def bulk_archive_cards(
    request: BulkArchiveCardsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkArchiveCardsResponse:
    """
    Archive multiple cards.

    - **card_ids**: List of card IDs to archive
    """
    try:
        result = db.bulk_archive_cards(card_ids=request.card_ids, archive=True)
        return BulkArchiveCardsResponse(
            success=result["success"],
            archived_count=result.get("archived_count", 0)
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk archive cards") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/cards/bulk-unarchive",
    response_model=BulkUnarchiveCardsResponse,
    summary="Bulk unarchive cards",
    description="Unarchive multiple cards.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.bulk"))]
)
async def bulk_unarchive_cards(
    request: BulkArchiveCardsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkUnarchiveCardsResponse:
    """
    Unarchive multiple cards.

    - **card_ids**: List of card IDs to unarchive
    """
    try:
        result = db.bulk_archive_cards(card_ids=request.card_ids, archive=False)
        return BulkUnarchiveCardsResponse(
            success=result["success"],
            unarchived_count=result.get("unarchived_count", 0)
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk unarchive cards") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/cards/bulk-delete",
    response_model=BulkDeleteCardsResponse,
    summary="Bulk delete cards",
    description="Soft delete multiple cards.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.bulk"))]
)
async def bulk_delete_cards(
    request: BulkDeleteCardsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkDeleteCardsResponse:
    """
    Soft delete multiple cards.

    - **card_ids**: List of card IDs to delete
    """
    try:
        result = db.bulk_delete_cards(card_ids=request.card_ids, hard_delete=False)
        return BulkDeleteCardsResponse(
            success=result["success"],
            deleted_count=result["deleted_count"]
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk delete cards") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/cards/bulk-label",
    response_model=BulkLabelCardsResponse,
    summary="Bulk label cards",
    description="Add and/or remove labels from multiple cards.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.bulk"))]
)
async def bulk_label_cards(
    request: BulkLabelCardsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BulkLabelCardsResponse:
    """
    Add and/or remove labels from multiple cards.

    - **card_ids**: List of card IDs to update
    - **add_label_ids**: Label IDs to add to all cards
    - **remove_label_ids**: Label IDs to remove from all cards
    """
    try:
        result = db.bulk_label_cards(
            card_ids=request.card_ids,
            add_label_ids=request.add_label_ids,
            remove_label_ids=request.remove_label_ids
        )
        return BulkLabelCardsResponse(
            success=result["success"],
            updated_count=result["updated_count"]
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to bulk label cards") from e
    except Exception as e:
        raise _handle_error(e) from e
# =============================================================================
# Card Filtering Endpoint (Phase 3)
# =============================================================================

@router.get(
    "/boards/{board_id}/cards",
    response_model=FilteredCardsResponse,
    summary="Get filtered cards for a board",
    description="Get all cards in a board with optional filters.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.filter"))]
)
async def get_filtered_cards(
    board_id: int,
    label_ids: Optional[str] = Query(None, description="Comma-separated label IDs (cards must have ALL)"),
    priority: Optional[str] = Query(None, description="Filter by priority (low, medium, high, urgent)"),
    due_before: Optional[str] = Query(None, description="Filter by due date before (ISO timestamp)"),
    due_after: Optional[str] = Query(None, description="Filter by due date after (ISO timestamp)"),
    overdue: Optional[bool] = Query(None, description="Only overdue cards (due_date < now AND not complete)"),
    has_due_date: Optional[bool] = Query(None, description="True for cards with due date, False for without"),
    has_checklist: Optional[bool] = Query(None, description="True for cards with checklists"),
    is_complete: Optional[bool] = Query(None, description="True for cards with all checklist items checked"),
    include_archived: bool = Query(False, description="Include archived cards"),
    include_deleted: bool = Query(False, description="Include soft-deleted cards"),
    limit: int = Query(50, ge=1, le=100, description="Maximum cards to return"),
    offset: int = Query(0, ge=0, description="Number of cards to skip"),
    page: Optional[int] = Query(None, ge=1, description="Legacy page number (1-indexed)"),
    per_page: Optional[int] = Query(None, ge=1, le=100, description="Legacy page size"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> FilteredCardsResponse:
    """
    Get filtered cards for a board.

    Supports filtering by labels, priority, due dates, checklist status, and more.
    """
    try:
        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)
        # Parse comma-separated label IDs
        parsed_label_ids = None
        if label_ids:
            parsed_label_ids = [int(lid.strip()) for lid in label_ids.split(",")]

        cards, total = db.get_board_cards_filtered(
            board_id=board_id,
            label_ids=parsed_label_ids,
            priority=priority,
            due_before=due_before,
            due_after=due_after,
            overdue=overdue,
            has_due_date=has_due_date,
            has_checklist=has_checklist,
            is_complete=is_complete,
            include_archived=include_archived,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset
        )

        return FilteredCardsResponse(
            cards=[CardResponse(**c) for c in cards],
            pagination=PaginationInfo(
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(cards)) < total
            )
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch filtered cards") from e
    except Exception as e:
        raise _handle_error(e) from e
# =============================================================================
# Enhanced Card Copy Endpoint (Phase 3)
# =============================================================================

@router.post(
    "/cards/{card_id}/copy-with-checklists",
    response_model=CardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Copy a card with checklists",
    description="Copy a card to a list, optionally including checklists and labels.",
    dependencies=[Depends(kanban_rate_limit("kanban.cards.copy"))]
)
async def copy_card_with_checklists(
    card_id: int,
    request: CardCopyWithChecklistsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> CardResponse:
    """
    Copy a card with checklists and labels.

    - **target_list_id**: Destination list ID (must be in same board)
    - **new_client_id**: Client-generated unique ID for the copy
    - **position**: Optional position in target list
    - **new_title**: Optional override title (defaults to "Copy of {original}")
    - **copy_checklists**: Whether to copy checklists (default True)
    - **copy_labels**: Whether to copy labels (default True)
    """
    try:
        card = db.copy_card_with_checklists(
            card_id=card_id,
            target_list_id=request.target_list_id,
            new_client_id=request.new_client_id,
            position=request.position,
            new_title=request.new_title,
            copy_checklists=request.copy_checklists,
            copy_labels=request.copy_labels
        )
        logger.info(f"Copied card {card_id} to {request.target_list_id} with checklists={request.copy_checklists}")
        return CardResponse(**card)
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to copy card") from e
    except Exception as e:
        raise _handle_error(e) from e
