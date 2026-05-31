# app/api/v1/endpoints/kanban_boards.py
"""
Kanban Board API endpoints.

Provides CRUD operations for Kanban boards including:
- Create, read, update boards
- Archive/unarchive boards
- Soft delete and restore boards
- Get board with nested lists and cards
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
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
    BoardCreate,
    BoardExportRequest,
    BoardExportResponse,
    BoardImportRequest,
    BoardImportResponse,
    BoardListResponse,
    BoardResponse,
    BoardUpdate,
    BoardWithListsResponse,
    DetailResponse,
    ImportStatsResponse,
    PaginationInfo,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)

router = APIRouter(prefix="/boards", tags=["Kanban Boards"])

def _export_board_data(
    db: KanbanDB,
    board_id: int,
    *,
    include_archived: bool,
    include_deleted: bool,
) -> BoardExportResponse:
    """Build a board export response with archive/delete options applied."""
    export_data = db.export_board(
        board_id=board_id,
        include_archived=include_archived,
        include_deleted=include_deleted,
    )
    return BoardExportResponse(**export_data)


# =============================================================================
# Board CRUD Endpoints
# =============================================================================

@router.post(
    "",
    response_model=BoardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new board",
    description="Create a new Kanban board for the authenticated user.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.create"))]
)
async def create_board(
    board_in: BoardCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardResponse:
    """
    Create a new Kanban board.

    - **name**: Board name (required, 1-255 characters)
    - **description**: Optional board description
    - **client_id**: Client-generated unique ID for idempotency
    - **activity_retention_days**: Optional retention period for activity logs (7-365 days)
    - **metadata**: Optional JSON metadata
    """
    try:
        board = db.create_board(
            name=board_in.name,
            client_id=board_in.client_id,
            description=board_in.description,
            activity_retention_days=board_in.activity_retention_days,
            metadata=board_in.metadata
        )
        logger.info(f"Created board {board['id']} for user {db.user_id}")
        return BoardResponse(**board)
    except (InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to create board") from e
@router.get(
    "",
    response_model=BoardListResponse,
    summary="List all boards",
    description="Get a paginated list of boards for the authenticated user.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.list"))]
)
async def list_boards(
    include_archived: bool = Query(False, description="Include archived boards"),
    include_deleted: bool = Query(False, description="Include soft-deleted boards"),
    limit: int = Query(50, ge=1, le=200, description="Maximum boards to return"),
    offset: int = Query(0, ge=0, description="Number of boards to skip"),
    page: Optional[int] = Query(None, ge=1, description="Legacy page number (1-indexed)"),
    per_page: Optional[int] = Query(None, ge=1, le=200, description="Legacy page size"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardListResponse:
    """
    List all boards for the authenticated user.

    Results are paginated and ordered by last update time (most recent first).
    """
    try:
        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)
        boards, total = db.list_boards(
            include_archived=include_archived,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset
        )
        return BoardListResponse(
            boards=[BoardResponse(**b) for b in boards],
            pagination=PaginationInfo(
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(boards)) < total
            )
        )
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch boards") from e
@router.get(
    "/{board_id}",
    response_model=BoardWithListsResponse,
    summary="Get a board",
    description="Get a board by ID, optionally including nested lists and cards.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.get"))]
)
async def get_board(
    board_id: int,
    include_lists: bool = Query(True, description="Include lists in response"),
    include_cards: bool = Query(True, description="Include cards in each list"),
    include_archived: bool = Query(False, description="Include archived items"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardWithListsResponse:
    """
    Get a board by ID.

    By default, returns the board with all its lists and cards nested.
    Use query parameters to control what's included.
    """
    try:
        if include_lists and include_cards:
            board = db.get_board_with_lists_and_cards(board_id, include_archived=include_archived)
        else:
            board = db.get_board(board_id)

        if not board:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board {board_id} not found"
            )

        if "labels" not in board:
            board["labels"] = db.list_labels(board_id)

        # If we didn't get nested data, add empty lists
        if "lists" not in board:
            board["lists"] = []
            board["total_cards"] = 0
        elif include_lists and not include_cards:
            # Get lists without cards
            lists = db.list_lists(board_id, include_archived=include_archived)
            for lst in lists:
                lst["cards"] = []
                lst["card_count"] = db.get_card_count_for_list(lst["id"])
            board["lists"] = lists
            board["total_cards"] = sum(lst["card_count"] for lst in lists)

        return BoardWithListsResponse(**board)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch board") from e


@router.patch(
    "/{board_id}",
    response_model=BoardResponse,
    summary="Update a board",
    description="Update board properties.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.update"))]
)
async def update_board(
    board_id: int,
    board_in: BoardUpdate,
    x_expected_version: Optional[int] = Header(None, description="Expected version for optimistic locking"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardResponse:
    """
    Update a board.

    All fields are optional. Only provided fields will be updated.
    Use the X-Expected-Version header for optimistic locking.
    """
    try:
        board = db.update_board(
            board_id=board_id,
            name=board_in.name,
            description=board_in.description,
            activity_retention_days=board_in.activity_retention_days,
            metadata=board_in.metadata,
            expected_version=x_expected_version
        )
        logger.info(f"Updated board {board_id}")
        return BoardResponse(**board)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to update board") from e
# =============================================================================
# Archive Operations
# =============================================================================

@router.post(
    "/{board_id}/archive",
    response_model=BoardResponse,
    summary="Archive a board",
    description="Archive a board. Archived boards are hidden from default listings but can be restored.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.archive"))]
)
async def archive_board(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardResponse:
    """Archive a board."""
    try:
        board = db.archive_board(board_id, archive=True)
        logger.info(f"Archived board {board_id}")
        return BoardResponse(**board)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to archive board") from e


@router.post(
    "/{board_id}/unarchive",
    response_model=BoardResponse,
    summary="Unarchive a board",
    description="Restore an archived board to active status.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.archive"))]
)
async def unarchive_board(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardResponse:
    """Unarchive a board."""
    try:
        board = db.archive_board(board_id, archive=False)
        logger.info(f"Unarchived board {board_id}")
        return BoardResponse(**board)
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to unarchive board") from e


# =============================================================================
# Delete Operations
# =============================================================================

@router.delete(
    "/{board_id}",
    response_model=DetailResponse,
    summary="Delete a board",
    description="Soft-delete a board. The board can be restored within the retention period.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.delete"))]
)
async def delete_board(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """
    Soft-delete a board.

    The board and all its lists and cards will be hidden but can be restored.
    """
    try:
        success = db.delete_board(board_id, hard_delete=False)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board {board_id} not found"
            )
        logger.info(f"Deleted board {board_id}")
        return DetailResponse(detail=f"Board {board_id} deleted successfully")
    except KanbanDBError as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete board") from e
@router.post(
    "/{board_id}/restore",
    response_model=BoardResponse,
    summary="Restore a deleted board",
    description="Restore a soft-deleted board.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.delete"))]
)
async def restore_board(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardResponse:
    """Restore a soft-deleted board."""
    try:
        board = db.restore_board(board_id)
        logger.info(f"Restored board {board_id}")
        return BoardResponse(**board)
    except (NotFoundError, InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to restore board") from e
# =============================================================================
# Activity Endpoints (Phase 2 placeholder)
# =============================================================================

@router.get(
    "/{board_id}/activities",
    response_model=ActivitiesListResponse,
    summary="Get board activities",
    description="Get activity log for a board.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.get"))]
)
async def get_board_activities(
    board_id: int,
    list_id: Optional[int] = Query(None, description="Filter by list ID"),
    card_id: Optional[int] = Query(None, description="Filter by card ID"),
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
    """
    Get activity log for a board.

    Activities track changes to boards, lists, and cards.
    """
    if created_after and created_before and created_after > created_before:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="created_after must be less than or equal to created_before.",
        )
    try:
        limit, offset = resolve_limit_offset(limit=limit, offset=offset, page=page, per_page=per_page)
        activities, total = db.get_board_activities(
            board_id=board_id,
            list_id=list_id,
            card_id=card_id,
            created_after=to_db_timestamp(created_after),
            created_before=to_db_timestamp(created_before),
            action_type=action_type,
            entity_type=entity_type,
            limit=limit,
            offset=offset
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
        raise map_db_error_to_http(e, default_detail="Failed to fetch board activities") from e
# =============================================================================
# Export/Import Endpoints (Phase 3)
# =============================================================================

@router.get(
    "/{board_id}/export",
    response_model=BoardExportResponse,
    summary="Export a board",
    description="Export a board with all its data as JSON.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.export"))]
)
async def export_board_get(
    board_id: int,
    include_archived: bool = Query(False, description="Include archived items in export"),
    include_deleted: bool = Query(False, description="Include soft-deleted items in export"),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardExportResponse:
    """Export a board with all its data."""
    try:
        return _export_board_data(
            db,
            board_id,
            include_archived=include_archived,
            include_deleted=include_deleted,
        )
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to export board") from e

@router.post(
    "/{board_id}/export",
    response_model=BoardExportResponse,
    summary="Export a board",
    description="Export a board with all its data as JSON.",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.export"))]
)
async def export_board(
    board_id: int,
    export_request: BoardExportRequest = BoardExportRequest(),
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardExportResponse:
    """
    Export a board with all its data.

    Exports the board with lists, cards, labels, checklists, and comments.
    The exported format is `tldw_kanban_v1` which can be re-imported.

    - **include_archived**: Include archived items in export (default: false)
    - **include_deleted**: Include soft-deleted items in export (default: false)
    """
    try:
        return _export_board_data(
            db,
            board_id,
            include_archived=export_request.include_archived,
            include_deleted=export_request.include_deleted,
        )
    except (NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to export board") from e
@router.post(
    "/import",
    response_model=BoardImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import a board",
    description="Import a board from JSON data (tldw_kanban_v1 or Trello format).",
    dependencies=[Depends(kanban_rate_limit("kanban.boards.import"))]
)
async def import_board(
    import_request: BoardImportRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> BoardImportResponse:
    """
    Import a board from JSON data.

    Supports two formats:
    - **tldw_kanban_v1**: Native format from board export
    - **Trello JSON**: Export from Trello board

    The format is auto-detected based on the data structure.

    - **data**: The board data to import (required)
    - **board_name**: Override name for the imported board (optional)
    """
    try:
        result = db.import_board(
            data=import_request.data,
            board_name=import_request.board_name
        )
        return BoardImportResponse(
            board=BoardResponse(**result["board"]),
            import_stats=ImportStatsResponse(**result["import_stats"])
        )
    except (InputError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to import board") from e
