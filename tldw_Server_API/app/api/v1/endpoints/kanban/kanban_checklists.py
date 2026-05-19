# app/api/v1/endpoints/kanban_checklists.py
"""
Kanban Checklist API endpoints.

Provides CRUD operations for Kanban checklists and checklist items including:
- Create, read, update, delete checklists
- Create, read, update, delete checklist items
- Reorder checklists and items
- Check/uncheck items
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
    handle_kanban_db_error,
    kanban_rate_limit,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.kanban_schemas import (
    ChecklistCreate,
    ChecklistItemCreate,
    ChecklistItemReorderRequest,
    ChecklistItemResponse,
    ChecklistItemsListResponse,
    ChecklistItemUpdate,
    ChecklistReorderRequest,
    ChecklistResponse,
    ChecklistsListResponse,
    ChecklistUpdate,
    ChecklistWithItemsResponse,
    ToggleAllChecklistItemsRequest,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)

router = APIRouter(tags=["Kanban Checklists"])


# --- Helper for Exception Handling ---
def _handle_error(e: Exception) -> HTTPException:
    """Convert exceptions to appropriate HTTP responses."""
    return handle_kanban_db_error(e)


# =============================================================================
# Checklist CRUD Endpoints
# =============================================================================

@router.post(
    "/cards/{card_id}/checklists",
    response_model=ChecklistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new checklist",
    description="Create a new checklist for a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklists.create"))]
)
async def create_checklist(
    card_id: int,
    checklist_in: ChecklistCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistResponse:
    """
    Create a new checklist for a card.

    - **name**: Checklist name (required, 1-255 characters)
    - **position**: Optional position (auto-assigned if not provided)
    """
    try:
        checklist = db.create_checklist(
            card_id=card_id,
            name=checklist_in.name,
            position=checklist_in.position
        )
        return ChecklistResponse(**checklist)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to create checklist") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/cards/{card_id}/checklists",
    response_model=ChecklistsListResponse,
    summary="List card checklists",
    description="Get all checklists for a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklists.list"))]
)
async def list_checklists(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistsListResponse:
    """Get all checklists for a card, ordered by position."""
    try:
        checklists = db.list_checklists(card_id=card_id)
        return ChecklistsListResponse(
            checklists=[ChecklistResponse(**cl) for cl in checklists]
        )
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch checklists") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/checklists/{checklist_id}",
    response_model=ChecklistWithItemsResponse,
    summary="Get a checklist",
    description="Get a checklist by ID with its items.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklists.get"))]
)
async def get_checklist(
    checklist_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistWithItemsResponse:
    """Get a checklist by ID with all its items and progress info."""
    try:
        checklist = db.get_checklist_with_items(checklist_id=checklist_id)
        if not checklist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checklist {checklist_id} not found"
            )
        return ChecklistWithItemsResponse(**checklist)
    except HTTPException:
        raise
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch checklist") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.patch(
    "/checklists/{checklist_id}",
    response_model=ChecklistResponse,
    summary="Update a checklist",
    description="Update an existing checklist.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklists.update"))]
)
async def update_checklist(
    checklist_id: int,
    checklist_in: ChecklistUpdate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistResponse:
    """
    Update a checklist.

    - **name**: New checklist name (optional)
    """
    try:
        checklist = db.update_checklist(
            checklist_id=checklist_id,
            name=checklist_in.name
        )
        return ChecklistResponse(**checklist)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to update checklist") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.delete(
    "/checklists/{checklist_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a checklist",
    description="Delete a checklist (hard delete). This deletes all items in the checklist.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklists.delete"))]
)
async def delete_checklist(
    checklist_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> Response:
    """Delete a checklist and all its items permanently."""
    try:
        success = db.delete_checklist(checklist_id=checklist_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checklist {checklist_id} not found"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete checklist") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/cards/{card_id}/checklists/reorder",
    response_model=ChecklistsListResponse,
    summary="Reorder checklists",
    description="Reorder checklists on a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklists.reorder"))]
)
async def reorder_checklists(
    card_id: int,
    reorder_in: ChecklistReorderRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistsListResponse:
    """
    Reorder checklists on a card.

    - **checklist_ids**: List of checklist IDs in the desired order
    """
    try:
        checklists = db.reorder_checklists(
            card_id=card_id,
            checklist_ids=reorder_in.checklist_ids
        )
        return ChecklistsListResponse(
            checklists=[ChecklistResponse(**cl) for cl in checklists]
        )
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to reorder checklists") from e
    except Exception as e:
        raise _handle_error(e) from e
# =============================================================================
# Checklist Item CRUD Endpoints
# =============================================================================

@router.post(
    "/checklists/{checklist_id}/items",
    response_model=ChecklistItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a checklist item",
    description="Create a new item in a checklist.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.create"))]
)
async def create_checklist_item(
    checklist_id: int,
    item_in: ChecklistItemCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemResponse:
    """
    Create a new item in a checklist.

    - **name**: Item name (required, 1-500 characters)
    - **position**: Optional position (auto-assigned if not provided)
    - **checked**: Whether the item starts checked (default: false)
    """
    try:
        item = db.create_checklist_item(
            checklist_id=checklist_id,
            name=item_in.name,
            position=item_in.position,
            checked=item_in.checked
        )
        return ChecklistItemResponse(**item)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to create checklist item") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/checklists/{checklist_id}/items",
    response_model=ChecklistItemsListResponse,
    summary="List checklist items",
    description="Get all items in a checklist.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.list"))]
)
async def list_checklist_items(
    checklist_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemsListResponse:
    """Get all items in a checklist, ordered by position."""
    try:
        items = db.list_checklist_items(checklist_id=checklist_id)
        return ChecklistItemsListResponse(
            items=[ChecklistItemResponse(**item) for item in items]
        )
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch checklist items") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/checklist-items/{item_id}",
    response_model=ChecklistItemResponse,
    summary="Get a checklist item",
    description="Get a checklist item by ID.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.get"))]
)
async def get_checklist_item(
    item_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemResponse:
    """Get a single checklist item by ID."""
    try:
        item = db.get_checklist_item(item_id=item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checklist item {item_id} not found"
            )
        return ChecklistItemResponse(**item)
    except HTTPException:
        raise
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch checklist item") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.patch(
    "/checklist-items/{item_id}",
    response_model=ChecklistItemResponse,
    summary="Update a checklist item",
    description="Update a checklist item (name and/or checked status).",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.update"))]
)
async def update_checklist_item(
    item_id: int,
    item_in: ChecklistItemUpdate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemResponse:
    """
    Update a checklist item.

    - **name**: New item name (optional)
    - **checked**: New checked status (optional)
    """
    try:
        item = db.update_checklist_item(
            item_id=item_id,
            name=item_in.name,
            checked=item_in.checked
        )
        return ChecklistItemResponse(**item)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to update checklist item") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.delete(
    "/checklist-items/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a checklist item",
    description="Delete a checklist item (hard delete).",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.delete"))]
)
async def delete_checklist_item(
    item_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> Response:
    """Delete a checklist item permanently."""
    try:
        success = db.delete_checklist_item(item_id=item_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checklist item {item_id} not found"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete checklist item") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/checklists/{checklist_id}/items/reorder",
    response_model=ChecklistItemsListResponse,
    summary="Reorder checklist items",
    description="Reorder items in a checklist.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.reorder"))]
)
async def reorder_checklist_items(
    checklist_id: int,
    reorder_in: ChecklistItemReorderRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemsListResponse:
    """
    Reorder items in a checklist.

    - **item_ids**: List of item IDs in the desired order
    """
    try:
        items = db.reorder_checklist_items(
            checklist_id=checklist_id,
            item_ids=reorder_in.item_ids
        )
        return ChecklistItemsListResponse(
            items=[ChecklistItemResponse(**item) for item in items]
        )
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to reorder checklist items") from e
    except Exception as e:
        raise _handle_error(e) from e
# =============================================================================
# Convenience Endpoints
# =============================================================================

@router.post(
    "/checklist-items/{item_id}/check",
    response_model=ChecklistItemResponse,
    summary="Check an item",
    description="Mark a checklist item as checked.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.update"))]
)
async def check_item(
    item_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemResponse:
    """Mark a checklist item as checked."""
    try:
        item = db.update_checklist_item(item_id=item_id, checked=True)
        return ChecklistItemResponse(**item)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to check checklist item") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/checklist-items/{item_id}/uncheck",
    response_model=ChecklistItemResponse,
    summary="Uncheck an item",
    description="Mark a checklist item as unchecked.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.update"))]
)
async def uncheck_item(
    item_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistItemResponse:
    """Mark a checklist item as unchecked."""
    try:
        item = db.update_checklist_item(item_id=item_id, checked=False)
        return ChecklistItemResponse(**item)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to uncheck checklist item") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.post(
    "/checklists/{checklist_id}/toggle-all",
    response_model=ChecklistWithItemsResponse,
    summary="Toggle all checklist items",
    description="Check or uncheck all items in a checklist.",
    dependencies=[Depends(kanban_rate_limit("kanban.checklist_items.toggle_all"))]
)
async def toggle_all_checklist_items(
    checklist_id: int,
    request: ToggleAllChecklistItemsRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> ChecklistWithItemsResponse:
    """
    Check or uncheck all items in a checklist.

    - **checked**: True to check all items, False to uncheck all items
    """
    try:
        checklist = db.toggle_all_checklist_items(
            checklist_id=checklist_id,
            checked=request.checked
        )
        return ChecklistWithItemsResponse(**checklist)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to toggle checklist items") from e
    except Exception as e:
        raise _handle_error(e) from e
