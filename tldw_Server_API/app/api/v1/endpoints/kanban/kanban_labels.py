# app/api/v1/endpoints/kanban_labels.py
"""
Kanban Label API endpoints.

Provides CRUD operations for Kanban labels including:
- Create, read, update, delete labels
- Assign and remove labels from cards
- Get labels for a board or card
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
    handle_kanban_db_error,
    kanban_rate_limit,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.kanban_schemas import (
    DetailResponse,
    LabelCreate,
    LabelResponse,
    LabelsListResponse,
    LabelUpdate,
)
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)

router = APIRouter(tags=["Kanban Labels"])


# --- Helper for Exception Handling ---
def _handle_error(e: Exception) -> HTTPException:
    """Convert exceptions to appropriate HTTP responses."""
    return handle_kanban_db_error(e)


# =============================================================================
# Label CRUD Endpoints (Board-scoped)
# =============================================================================

@router.post(
    "/boards/{board_id}/labels",
    response_model=LabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new label",
    description="Create a new label for a board.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.create"))]
)
async def create_label(
    board_id: int,
    label_in: LabelCreate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> LabelResponse:
    """
    Create a new label for a board.

    - **name**: Label name (required, 1-50 characters)
    - **color**: Label color (red, orange, yellow, green, blue, purple, pink, gray)
    """
    try:
        label = db.create_label(
            board_id=board_id,
            name=label_in.name,
            color=label_in.color
        )
        return LabelResponse(**label)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to create label") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/boards/{board_id}/labels",
    response_model=LabelsListResponse,
    summary="List board labels",
    description="Get all labels for a board.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.list"))]
)
async def list_labels(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> LabelsListResponse:
    """Get all labels for a board, ordered alphabetically."""
    try:
        labels = db.list_labels(board_id=board_id)
        return LabelsListResponse(
            labels=[LabelResponse(**label) for label in labels]
        )
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch labels") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/labels/{label_id}",
    response_model=LabelResponse,
    summary="Get a label",
    description="Get a label by ID.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.get"))]
)
async def get_label(
    label_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> LabelResponse:
    """Get a single label by ID."""
    try:
        label = db.get_label(label_id=label_id)
        if not label:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label {label_id} not found"
            )
        return LabelResponse(**label)
    except HTTPException:
        raise
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch label") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.patch(
    "/labels/{label_id}",
    response_model=LabelResponse,
    summary="Update a label",
    description="Update an existing label.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.update"))]
)
async def update_label(
    label_id: int,
    label_in: LabelUpdate,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> LabelResponse:
    """
    Update a label.

    - **name**: New label name (optional)
    - **color**: New label color (optional)
    """
    try:
        label = db.update_label(
            label_id=label_id,
            name=label_in.name,
            color=label_in.color
        )
        return LabelResponse(**label)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to update label") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.delete(
    "/labels/{label_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a label",
    description="Delete a label (hard delete). This removes the label from all cards.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.delete"))]
)
async def delete_label(
    label_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> Response:
    """Delete a label permanently."""
    try:
        success = db.delete_label(label_id=label_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label {label_id} not found"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to delete label") from e
    except Exception as e:
        raise _handle_error(e) from e
# =============================================================================
# Card-Label Association Endpoints
# =============================================================================

@router.post(
    "/cards/{card_id}/labels/{label_id}",
    response_model=DetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Assign label to card",
    description="Assign a label to a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.assign"))]
)
async def assign_label_to_card(
    card_id: int,
    label_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> DetailResponse:
    """
    Assign a label to a card.

    The label must belong to the same board as the card.
    If the label is already assigned, this is a no-op.
    """
    try:
        db.assign_label_to_card(card_id=card_id, label_id=label_id)
        return DetailResponse(detail="Label assigned to card")
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to assign label to card") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.delete(
    "/cards/{card_id}/labels/{label_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove label from card",
    description="Remove a label from a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.remove"))]
)
async def remove_label_from_card(
    card_id: int,
    label_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> Response:
    """Remove a label from a card."""
    try:
        db.remove_label_from_card(card_id=card_id, label_id=label_id)
        # Don't raise 404 if the association didn't exist - idempotent behavior
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to remove label from card") from e
    except Exception as e:
        raise _handle_error(e) from e
@router.get(
    "/cards/{card_id}/labels",
    response_model=LabelsListResponse,
    summary="Get card labels",
    description="Get all labels assigned to a card.",
    dependencies=[Depends(kanban_rate_limit("kanban.labels.list"))]
)
async def get_card_labels(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user)
) -> LabelsListResponse:
    """Get all labels assigned to a card."""
    try:
        labels = db.get_card_labels(card_id=card_id)
        return LabelsListResponse(
            labels=[LabelResponse(**label) for label in labels]
        )
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as e:
        raise map_db_error_to_http(e, default_detail="Failed to fetch card labels") from e
    except Exception as e:
        raise _handle_error(e) from e
