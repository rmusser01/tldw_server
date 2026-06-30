# app/api/v1/endpoints/kanban_lists.py
"""
Kanban List API endpoints.

Provides CRUD operations for Kanban lists including:
- Create, read, update lists
- Archive/unarchive lists
- Soft delete and restore lists
- Reorder lists within a board
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
    DetailResponse,
    ListCreate,
    ListResponse,
    ListsListResponse,
    ListUpdate,
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

router = APIRouter(tags=["Kanban Lists"])


# --- Helper for Exception Handling ---
def _handle_error(e: Exception) -> HTTPException:
    """Convert exceptions to appropriate HTTP responses."""
    return handle_kanban_db_error(e)


# =============================================================================
# List CRUD Endpoints (nested under /boards/{board_id}/lists)
# =============================================================================

@router.post(
    "/boards/{board_id}/lists",
    response_model=ListResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new list",
    description="Create a new list in a board.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.create"))]
)
async def create_list(
    board_id: int,
    list_in: ListCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListResponse:
    """
    Create a new list in a board.

    - **name**: List name (required, 1-255 characters)
    - **client_id**: Client-generated unique ID for idempotency
    - **position**: Optional position (0-indexed, defaults to end)
    """
    try:
        lst = db.create_list(
            board_id=board_id,
            name=list_in.name,
            client_id=list_in.client_id,
            position=list_in.position
        )
        lst["card_count"] = 0  # New list has no cards
        logger.info(f"Created list {lst['id']} in board {board_id}")
        return ListResponse(**lst)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to create list") from e
@router.get(
    "/boards/{board_id}/lists",
    response_model=ListsListResponse,
    summary="Get all lists in a board",
    description="Get all lists for a board, ordered by position.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.list"))]
)
async def get_lists(
    board_id: int,
    include_archived: bool = Query(False, description="Include archived lists"),
    include_deleted: bool = Query(False, description="Include soft-deleted lists"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListsListResponse:
    """
    Get all lists in a board.

    Lists are returned ordered by position (ascending).
    """
    try:
        lists = db.list_lists(
            board_id=board_id,
            include_archived=include_archived,
            include_deleted=include_deleted
        )
        # Add card count to each list
        for lst in lists:
            lst["card_count"] = db.get_card_count_for_list(lst["id"])
        return ListsListResponse(lists=[ListResponse(**lst) for lst in lists])
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch lists") from e
@router.post(
    "/boards/{board_id}/lists/reorder",
    response_model=ReorderResponse,
    summary="Reorder lists in a board",
    description="Set the order of lists in a board.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.reorder"))]
)
async def reorder_lists(
    board_id: int,
    reorder_in: ReorderRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ReorderResponse:
    """
    Reorder lists in a board.

    Provide the list IDs in the desired order. All active (non-deleted) lists
    in the board must be included.
    """
    try:
        ids = reorder_in.ids or []
        db.reorder_lists(board_id=board_id, list_ids=ids)
        logger.info(f"Reordered {len(ids)} lists in board {board_id}")
        return ReorderResponse(success=True, message="Lists reordered successfully")
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to reorder lists") from e
# =============================================================================
# Individual List Endpoints (at /lists/{list_id})
# =============================================================================

@router.get(
    "/lists/{list_id}",
    response_model=ListResponse,
    summary="Get a list",
    description="Get a list by ID.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.get"))]
)
async def get_list(
    list_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListResponse:
    """Get a list by ID."""
    try:
        lst = db.get_list(list_id)
        if not lst:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"List {list_id} not found"
            )
        lst["card_count"] = db.get_card_count_for_list(list_id)
        return ListResponse(**lst)
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch list") from e
@router.patch(
    "/lists/{list_id}",
    response_model=ListResponse,
    summary="Update a list",
    description="Update list properties.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.update"))]
)
async def update_list(
    list_id: int,
    list_in: ListUpdate,
    x_expected_version: Optional[int] = Header(None, description="Expected version for optimistic locking"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListResponse:
    """
    Update a list.

    All fields are optional. Only provided fields will be updated.
    Use the X-Expected-Version header for optimistic locking.
    """
    try:
        lst = db.update_list(
            list_id=list_id,
            name=list_in.name,
            position=list_in.position,
            expected_version=x_expected_version
        )
        lst["card_count"] = db.get_card_count_for_list(list_id)
        logger.info(f"Updated list {list_id}")
        return ListResponse(**lst)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to update list") from e
# =============================================================================
# Archive Operations
# =============================================================================

@router.post(
    "/lists/{list_id}/archive",
    response_model=ListResponse,
    summary="Archive a list",
    description="Archive a list. Archived lists are hidden from default listings but can be restored.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.update"))]
)
async def archive_list(
    list_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListResponse:
    """Archive a list."""
    try:
        lst = db.archive_list(list_id, archive=True)
        lst["card_count"] = db.get_card_count_for_list(list_id)
        logger.info(f"Archived list {list_id}")
        return ListResponse(**lst)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to archive list") from e
@router.post(
    "/lists/{list_id}/unarchive",
    response_model=ListResponse,
    summary="Unarchive a list",
    description="Restore an archived list to active status.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.update"))]
)
async def unarchive_list(
    list_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListResponse:
    """Unarchive a list."""
    try:
        lst = db.archive_list(list_id, archive=False)
        lst["card_count"] = db.get_card_count_for_list(list_id)
        logger.info(f"Unarchived list {list_id}")
        return ListResponse(**lst)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to unarchive list") from e
# =============================================================================
# Delete Operations
# =============================================================================

@router.delete(
    "/lists/{list_id}",
    response_model=DetailResponse,
    summary="Delete a list",
    description="Soft-delete a list. The list can be restored within the retention period.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.delete"))]
)
async def delete_list(
    list_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """
    Soft-delete a list.

    The list and all its cards will be hidden but can be restored.
    """
    try:
        success = db.delete_list(list_id, hard_delete=False)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"List {list_id} not found"
            )
        logger.info(f"Deleted list {list_id}")
        return DetailResponse(detail=f"List {list_id} deleted successfully")
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete list") from e
@router.post(
    "/lists/{list_id}/restore",
    response_model=ListResponse,
    summary="Restore a deleted list",
    description="Restore a soft-deleted list.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.delete"))]
)
async def restore_list(
    list_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ListResponse:
    """Restore a soft-deleted list."""
    try:
        lst = db.restore_list(list_id)
        lst["card_count"] = db.get_card_count_for_list(list_id)
        logger.info(f"Restored list {list_id}")
        return ListResponse(**lst)
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to restore list") from e
# =============================================================================
# Activity Endpoints
# =============================================================================

@router.get(
    "/lists/{list_id}/activities",
    response_model=ActivitiesListResponse,
    summary="Get list activities",
    description="Get activity log for a list.",
    dependencies=[Depends(kanban_rate_limit("kanban.lists.get"))]
)
async def get_list_activities(
    list_id: int,
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
    """Get activity log for a list."""
    if created_after and created_before and created_after > created_before:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="created_after must be less than or equal to created_before.",
        )
    try:
        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)
        activities, total = db.get_list_activities(
            list_id=list_id,
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
        raise map_db_error_to_http(e, default_detail="Failed to fetch list activities") from e
