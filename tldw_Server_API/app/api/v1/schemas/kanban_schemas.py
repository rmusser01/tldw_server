# app/api/v1/schemas/kanban_schemas.py
"""
Pydantic schemas for Kanban Board API endpoints.

Implements Base/Create/Update/Response pattern for:
- Boards
- Lists
- Cards

Phase 1 scope - additional schemas for labels, checklists, comments
will be added in Phase 2.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

# =============================================================================
# Common Schemas
# =============================================================================

class DetailResponse(BaseModel):
    """Standard error/detail response."""
    detail: str


class PaginationInfo(BaseModel):
    """Pagination metadata."""
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Maximum items per page")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether more items exist")


# =============================================================================
# Board Schemas
# =============================================================================

class BoardBase(BaseModel):
    """Base schema for board fields."""
    name: str = Field(..., min_length=1, max_length=255, description="Board name")
    description: Optional[str] = Field(None, max_length=5000, description="Board description")


class BoardCreate(BoardBase):
    """Schema for creating a new board."""
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Client-generated unique ID for idempotency"
    )
    activity_retention_days: Optional[int] = Field(
        None,
        ge=7,
        le=365,
        description="Activity log retention period in days (7-365)"
    )
    metadata: Optional[dict[str, Any]] = Field(None, description="Optional JSON metadata")


class BoardUpdate(BaseModel):
    """Schema for updating a board."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="New board name")
    description: Optional[str] = Field(None, max_length=5000, description="New board description")
    activity_retention_days: Optional[int] = Field(
        None,
        ge=7,
        le=365,
        description="Activity log retention period in days (7-365)"
    )
    metadata: Optional[dict[str, Any]] = Field(None, description="Optional JSON metadata")


class BoardResponse(BoardBase):
    """Schema for board response."""
    id: int = Field(..., description="Board ID")
    uuid: str = Field(..., description="Board UUID")
    user_id: str = Field(..., description="Owner user ID")
    client_id: str = Field(..., description="Client-generated ID")
    archived: bool = Field(..., description="Whether the board is archived")
    archived_at: Optional[datetime] = Field(None, description="When the board was archived")
    activity_retention_days: Optional[int] = Field(
        None,
        description="Activity log retention period in days"
    )
    created_at: datetime = Field(..., description="When the board was created")
    updated_at: datetime = Field(..., description="When the board was last updated")
    deleted: bool = Field(..., description="Whether the board is soft-deleted")
    deleted_at: Optional[datetime] = Field(None, description="When the board was deleted")
    version: int = Field(..., description="Version number for optimistic locking")
    metadata: Optional[dict[str, Any]] = Field(None, description="JSON metadata")
    list_count: Optional[int] = Field(None, description="Number of lists in the board")
    card_count: Optional[int] = Field(None, description="Number of cards in the board")

    model_config = ConfigDict(from_attributes=True)


class BoardListResponse(BaseModel):
    """Schema for paginated list of boards."""
    boards: list[BoardResponse] = Field(..., description="List of boards")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


# =============================================================================
# List Schemas
# =============================================================================

class ListBase(BaseModel):
    """Base schema for list fields."""
    name: str = Field(..., min_length=1, max_length=255, description="List name")


class ListCreate(ListBase):
    """Schema for creating a new list."""
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Client-generated unique ID for idempotency"
    )
    position: Optional[int] = Field(
        None,
        ge=0,
        description="Position in the board (0-indexed, defaults to end)"
    )


class ListUpdate(BaseModel):
    """Schema for updating a list."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="New list name")
    position: Optional[int] = Field(None, ge=0, description="New position in the board")


class ListResponse(ListBase):
    """Schema for list response."""
    id: int = Field(..., description="List ID")
    uuid: str = Field(..., description="List UUID")
    board_id: int = Field(..., description="Parent board ID")
    client_id: str = Field(..., description="Client-generated ID")
    position: int = Field(..., description="Position in the board")
    archived: bool = Field(..., description="Whether the list is archived")
    archived_at: Optional[datetime] = Field(None, description="When the list was archived")
    created_at: datetime = Field(..., description="When the list was created")
    updated_at: datetime = Field(..., description="When the list was last updated")
    deleted: bool = Field(..., description="Whether the list is soft-deleted")
    deleted_at: Optional[datetime] = Field(None, description="When the list was deleted")
    version: int = Field(..., description="Version number for optimistic locking")
    card_count: Optional[int] = Field(None, description="Number of cards in the list")

    model_config = ConfigDict(from_attributes=True)


class ListsListResponse(BaseModel):
    """Schema for list of lists (not paginated since typically small)."""
    lists: list[ListResponse] = Field(..., description="List of lists")


class ListPositionItem(BaseModel):
    """Legacy list reorder item."""
    list_id: int = Field(..., description="List ID")
    position: int = Field(..., ge=0, description="Position in board")


class CardPositionItem(BaseModel):
    """Legacy card reorder item."""
    card_id: int = Field(..., description="Card ID")
    position: int = Field(..., ge=0, description="Position in list")


class ReorderRequest(BaseModel):
    """Schema for reordering lists/cards with compatibility shims."""
    ids: Optional[list[int]] = Field(
        None,
        min_length=1,
        description="Item IDs in the desired order"
    )
    list_positions: Optional[list[ListPositionItem]] = Field(
        None,
        min_length=1,
        description="Legacy list reorder payload; converted to ids by position"
    )
    card_positions: Optional[list[CardPositionItem]] = Field(
        None,
        min_length=1,
        description="Legacy card reorder payload; converted to ids by position"
    )

    @model_validator(mode="after")
    def normalize_legacy_payloads(self) -> "ReorderRequest":
        if self.ids:
            return self

        if self.list_positions:
            ordered = sorted(self.list_positions, key=lambda item: item.position)
            self.ids = [item.list_id for item in ordered]
            return self

        if self.card_positions:
            ordered = sorted(self.card_positions, key=lambda item: item.position)
            self.ids = [item.card_id for item in ordered]
            return self

        raise ValueError("Provide one of: ids, list_positions, or card_positions")


class ReorderResponse(BaseModel):
    """Schema for reorder response."""
    success: bool = Field(..., description="Whether the reorder succeeded")
    message: Optional[str] = Field(None, description="Optional message")


# =============================================================================
# Card Schemas
# =============================================================================

PriorityType = Literal["low", "medium", "high", "urgent"]


class CardBase(BaseModel):
    """Base schema for card fields."""
    title: str = Field(..., min_length=1, max_length=500, description="Card title")
    description: Optional[str] = Field(None, max_length=50000, description="Card description")


class CardCreate(CardBase):
    """Schema for creating a new card."""
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Client-generated unique ID for idempotency"
    )
    position: Optional[int] = Field(
        None,
        ge=0,
        description="Position in the list (0-indexed, defaults to end)"
    )
    due_date: Optional[datetime] = Field(None, description="Due date for the card")
    start_date: Optional[datetime] = Field(None, description="Start date for the card")
    priority: Optional[PriorityType] = Field(
        None,
        description="Card priority: low, medium, high, or urgent"
    )
    label_ids: Optional[list[int]] = Field(
        None,
        description="Optional label IDs to assign after card creation"
    )
    metadata: Optional[dict[str, Any]] = Field(None, description="Optional JSON metadata")


class CardUpdate(BaseModel):
    """Schema for updating a card."""
    title: Optional[str] = Field(None, min_length=1, max_length=500, description="New card title")
    description: Optional[str] = Field(None, max_length=50000, description="New card description")
    due_date: Optional[datetime] = Field(None, description="New due date")
    due_complete: Optional[bool] = Field(None, description="Whether due date is marked complete")
    start_date: Optional[datetime] = Field(None, description="New start date")
    priority: Optional[PriorityType] = Field(
        None,
        description="New priority: low, medium, high, or urgent"
    )
    metadata: Optional[dict[str, Any]] = Field(None, description="New JSON metadata")


class CardResponse(CardBase):
    """Schema for card response."""
    id: int = Field(..., description="Card ID")
    uuid: str = Field(..., description="Card UUID")
    board_id: int = Field(..., description="Parent board ID")
    list_id: int = Field(..., description="Parent list ID")
    client_id: str = Field(..., description="Client-generated ID")
    position: int = Field(..., description="Position in the list")
    due_date: Optional[datetime] = Field(None, description="Due date")
    due_complete: bool = Field(..., description="Whether due date is marked complete")
    start_date: Optional[datetime] = Field(None, description="Start date")
    priority: Optional[PriorityType] = Field(None, description="Card priority")
    archived: bool = Field(..., description="Whether the card is archived")
    archived_at: Optional[datetime] = Field(None, description="When the card was archived")
    created_at: datetime = Field(..., description="When the card was created")
    updated_at: datetime = Field(..., description="When the card was last updated")
    deleted: bool = Field(..., description="Whether the card is soft-deleted")
    deleted_at: Optional[datetime] = Field(None, description="When the card was deleted")
    version: int = Field(..., description="Version number for optimistic locking")
    metadata: Optional[dict[str, Any]] = Field(None, description="JSON metadata")

    model_config = ConfigDict(from_attributes=True)


class CardsListResponse(BaseModel):
    """Schema for list of cards (not paginated since within a list)."""
    cards: list[CardResponse] = Field(..., description="List of cards")


class CardMoveRequest(BaseModel):
    """Schema for moving a card to a different list."""
    target_list_id: int = Field(..., description="Destination list ID")
    position: Optional[int] = Field(
        None,
        ge=0,
        description="Position in target list (defaults to end)"
    )


class CardCopyRequest(BaseModel):
    """Schema for copying a card."""
    target_list_id: int = Field(..., description="Destination list ID")
    new_client_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Client-generated unique ID for the copy"
    )
    position: Optional[int] = Field(
        None,
        ge=0,
        description="Position in target list (defaults to end)"
    )
    new_title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=500,
        description="New title for the copy (defaults to 'Copy of {original}')"
    )


# =============================================================================
# Search Schemas
# =============================================================================

class CardSearchRequest(BaseModel):
    """Schema for card search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    board_id: Optional[int] = Field(None, description="Filter by board ID")
    limit: int = Field(50, ge=1, le=200, description="Maximum results")
    offset: int = Field(0, ge=0, description="Results to skip")
    page: Optional[int] = Field(None, ge=1, description="Legacy page number (1-indexed)")
    per_page: Optional[int] = Field(None, ge=1, le=200, description="Legacy page size")

    @model_validator(mode="after")
    def normalize_legacy_pagination(self) -> "CardSearchRequest":
        if self.per_page is not None:
            self.limit = self.per_page
        if self.page is not None:
            self.offset = (self.page - 1) * self.limit
        return self


class CardSearchResponse(BaseModel):
    """Schema for card search response."""
    cards: list[CardResponse] = Field(..., description="Matching cards")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


# =============================================================================
# Activity Schemas (Phase 2, but needed for endpoint stubs)
# =============================================================================

class ActivityResponse(BaseModel):
    """Schema for activity log entry."""
    id: int = Field(..., description="Activity ID")
    uuid: str = Field(..., description="Activity UUID")
    board_id: int = Field(..., description="Board ID")
    list_id: Optional[int] = Field(None, description="Related list ID")
    card_id: Optional[int] = Field(None, description="Related card ID")
    user_id: str = Field(..., description="User who performed the action")
    action_type: str = Field(..., description="Type of action (create, update, delete, etc.)")
    entity_type: str = Field(..., description="Type of entity (board, list, card, etc.)")
    entity_id: Optional[int] = Field(None, description="ID of the affected entity")
    details: Optional[dict[str, Any]] = Field(None, description="Additional details")
    created_at: datetime = Field(..., description="When the activity occurred")

    model_config = ConfigDict(from_attributes=True)


class ActivitiesListResponse(BaseModel):
    """Schema for paginated list of activities."""
    activities: list[ActivityResponse] = Field(..., description="List of activities")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


# =============================================================================
# Bulk Operation Schemas
# =============================================================================

class BulkArchiveRequest(BaseModel):
    """Schema for bulk archive operations."""
    ids: list[int] = Field(..., min_length=1, max_length=100, description="IDs to archive")
    archive: bool = Field(True, description="True to archive, False to unarchive")


class BulkDeleteRequest(BaseModel):
    """Schema for bulk delete operations."""
    ids: list[int] = Field(..., min_length=1, max_length=100, description="IDs to delete")


class BulkOperationResult(BaseModel):
    """Schema for bulk operation result."""
    success: bool = Field(..., description="Whether the operation succeeded")
    processed: int = Field(..., description="Number of items processed")
    failed: int = Field(0, description="Number of items that failed")
    errors: Optional[list[str]] = Field(None, description="Error messages if any")


# =============================================================================
# Label Schemas (Phase 2)
# =============================================================================

# Valid label colors
LABEL_COLORS = Literal["red", "orange", "yellow", "green", "blue", "purple", "pink", "gray"]


class LabelBase(BaseModel):
    """Base schema for label fields."""
    name: str = Field(..., min_length=1, max_length=50, description="Label name")
    color: LABEL_COLORS = Field(..., description="Label color")


class LabelCreate(LabelBase):
    """Schema for creating a new label."""
    pass


class LabelUpdate(BaseModel):
    """Schema for updating a label."""
    name: Optional[str] = Field(None, min_length=1, max_length=50, description="New label name")
    color: Optional[LABEL_COLORS] = Field(None, description="New label color")


class LabelResponse(LabelBase):
    """Schema for label response."""
    id: int = Field(..., description="Label ID")
    uuid: str = Field(..., description="Label UUID")
    board_id: int = Field(..., description="Board ID this label belongs to")
    created_at: datetime = Field(..., description="When the label was created")
    updated_at: datetime = Field(..., description="When the label was last updated")

    model_config = ConfigDict(from_attributes=True)


class LabelsListResponse(BaseModel):
    """Schema for list of labels."""
    labels: list[LabelResponse] = Field(..., description="List of labels")


# =============================================================================
# Nested Response Schemas (for GET /boards/{id} with lists and cards)
# =============================================================================

class CardInListResponse(CardResponse):
    """Card response nested within a list."""
    labels: list[LabelResponse] = Field(default_factory=list, description="Labels assigned to this card")
    checklist_count: int = Field(0, description="Number of checklists on the card")
    checklist_complete: int = Field(0, description="Number of completed checklist items")
    checklist_total: int = Field(0, description="Total checklist items")
    comment_count: int = Field(0, description="Number of comments on the card")


class ListWithCardsResponse(ListResponse):
    """List response with nested cards."""
    cards: list[CardInListResponse] = Field(default_factory=list, description="Cards in this list")


class BoardWithListsResponse(BoardResponse):
    """Board response with nested lists and cards."""
    labels: list[LabelResponse] = Field(default_factory=list, description="Labels available on this board")
    lists: list[ListWithCardsResponse] = Field(default_factory=list, description="Lists in this board")
    total_cards: int = Field(0, description="Total number of cards across all lists")


# =============================================================================
# Checklist Schemas (Phase 2)
# =============================================================================

class ChecklistBase(BaseModel):
    """Base schema for checklist fields."""
    name: str = Field(..., min_length=1, max_length=255, description="Checklist name")


class ChecklistCreate(ChecklistBase):
    """Schema for creating a new checklist."""
    position: Optional[int] = Field(None, ge=0, description="Position in the card (auto-assigned if not provided)")


class ChecklistUpdate(BaseModel):
    """Schema for updating a checklist."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="New checklist name")


class ChecklistResponse(ChecklistBase):
    """Schema for checklist response."""
    id: int = Field(..., description="Checklist ID")
    uuid: str = Field(..., description="Checklist UUID")
    card_id: int = Field(..., description="Card ID this checklist belongs to")
    position: int = Field(..., description="Position in the card")
    created_at: datetime = Field(..., description="When the checklist was created")
    updated_at: datetime = Field(..., description="When the checklist was last updated")

    model_config = ConfigDict(from_attributes=True)


class ChecklistsListResponse(BaseModel):
    """Schema for list of checklists."""
    checklists: list[ChecklistResponse] = Field(..., description="List of checklists")


class ChecklistReorderRequest(BaseModel):
    """Schema for reordering checklists."""
    checklist_ids: list[int] = Field(
        ...,
        min_length=1,
        description="Checklist IDs in the desired order"
    )


# =============================================================================
# Checklist Item Schemas (Phase 2)
# =============================================================================

class ChecklistItemBase(BaseModel):
    """Base schema for checklist item fields."""
    name: str = Field(..., min_length=1, max_length=500, description="Item name")


class ChecklistItemCreate(ChecklistItemBase):
    """Schema for creating a new checklist item."""
    position: Optional[int] = Field(None, ge=0, description="Position in the checklist (auto-assigned if not provided)")
    checked: bool = Field(False, description="Whether the item starts checked")


class ChecklistItemUpdate(BaseModel):
    """Schema for updating a checklist item."""
    name: Optional[str] = Field(None, min_length=1, max_length=500, description="New item name")
    checked: Optional[bool] = Field(None, description="Whether the item is checked")


class ChecklistItemResponse(ChecklistItemBase):
    """Schema for checklist item response."""
    id: int = Field(..., description="Item ID")
    uuid: str = Field(..., description="Item UUID")
    checklist_id: int = Field(..., description="Checklist ID this item belongs to")
    position: int = Field(..., description="Position in the checklist")
    checked: bool = Field(..., description="Whether the item is checked")
    checked_at: Optional[datetime] = Field(None, description="When the item was checked")
    created_at: datetime = Field(..., description="When the item was created")
    updated_at: datetime = Field(..., description="When the item was last updated")

    model_config = ConfigDict(from_attributes=True)


class ChecklistItemsListResponse(BaseModel):
    """Schema for list of checklist items."""
    items: list[ChecklistItemResponse] = Field(..., description="List of items")


class ChecklistItemReorderRequest(BaseModel):
    """Schema for reordering checklist items."""
    item_ids: list[int] = Field(
        ...,
        min_length=1,
        description="Item IDs in the desired order"
    )


class ChecklistWithItemsResponse(ChecklistResponse):
    """Schema for checklist with items included."""
    items: list[ChecklistItemResponse] = Field(default_factory=list, description="Checklist items")
    total_items: int = Field(0, description="Total number of items")
    checked_items: int = Field(0, description="Number of checked items")
    progress_percent: int = Field(0, ge=0, le=100, description="Completion percentage")


# =============================================================================
# Comment Schemas (Phase 2)
# =============================================================================

class CommentBase(BaseModel):
    """Base schema for comment fields."""
    content: str = Field(..., min_length=1, max_length=10000, description="Comment content (markdown supported)")


class CommentCreate(CommentBase):
    """Schema for creating a new comment."""
    pass


class CommentUpdate(CommentBase):
    """Schema for updating a comment."""
    pass


class CommentResponse(CommentBase):
    """Schema for comment response."""
    id: int = Field(..., description="Comment ID")
    uuid: str = Field(..., description="Comment UUID")
    card_id: int = Field(..., description="Card ID this comment belongs to")
    user_id: str = Field(..., description="User who created the comment")
    created_at: datetime = Field(..., description="When the comment was created")
    updated_at: datetime = Field(..., description="When the comment was last updated")
    deleted: bool = Field(False, description="Whether the comment is soft-deleted")

    model_config = ConfigDict(from_attributes=True)


class CommentsListResponse(BaseModel):
    """Schema for paginated list of comments."""
    comments: list[CommentResponse] = Field(..., description="List of comments")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


# =============================================================================
# Extended Card Response with Phase 2 Features
# =============================================================================

class CardWithDetailsResponse(CardResponse):
    """Card response with labels, checklists, and comment count."""
    labels: list[LabelResponse] = Field(default_factory=list, description="Labels assigned to this card")
    checklists: list[ChecklistWithItemsResponse] = Field(default_factory=list, description="Checklists on this card")
    comment_count: int = Field(0, description="Number of comments on this card")


# =============================================================================
# Export/Import Schemas (Phase 3)
# =============================================================================

class BoardExportRequest(BaseModel):
    """Schema for board export request."""
    include_archived: bool = Field(False, description="Include archived items in export")
    include_deleted: bool = Field(False, description="Include soft-deleted items in export")


class BoardExportResponse(BaseModel):
    """Schema for exported board data (JSON format)."""
    format: str = Field(..., description="Export format identifier")
    exported_at: str = Field(..., description="ISO timestamp of export")
    board: dict[str, Any] = Field(..., description="Board data")
    labels: list[dict[str, Any]] = Field(..., description="Board labels")
    lists: list[dict[str, Any]] = Field(..., description="Lists with cards, checklists, comments")


class BoardImportRequest(BaseModel):
    """Schema for board import request."""
    data: dict[str, Any] = Field(..., description="Board data to import (tldw_kanban_v1 or Trello format)")
    board_name: Optional[str] = Field(None, description="Override name for imported board")


class ImportStatsResponse(BaseModel):
    """Statistics from board import."""
    board_id: int = Field(..., description="ID of the created board")
    lists_imported: int = Field(0, description="Number of lists imported")
    cards_imported: int = Field(0, description="Number of cards imported")
    labels_imported: int = Field(0, description="Number of labels imported")
    checklists_imported: int = Field(0, description="Number of checklists imported")
    checklist_items_imported: int = Field(0, description="Number of checklist items imported")
    comments_imported: int = Field(0, description="Number of comments imported")


class BoardImportResponse(BaseModel):
    """Schema for board import response."""
    board: BoardResponse = Field(..., description="The imported board")
    import_stats: ImportStatsResponse = Field(..., description="Import statistics")


# =============================================================================
# Bulk Operations Schemas (Phase 3)
# =============================================================================

class BulkMoveCardsRequest(BaseModel):
    """Schema for bulk move cards request."""
    card_ids: list[int] = Field(..., min_length=1, description="List of card IDs to move")
    target_list_id: int = Field(..., description="Destination list ID")
    position: Optional[int] = Field(None, ge=0, description="Starting position in target list")


class BulkMoveCardsResponse(BaseModel):
    """Schema for bulk move cards response."""
    success: bool = Field(..., description="Whether the operation succeeded")
    moved_count: int = Field(..., description="Number of cards moved")
    cards: list[CardResponse] = Field(..., description="Updated cards")


class BulkArchiveCardsRequest(BaseModel):
    """Schema for bulk archive cards request."""
    card_ids: list[int] = Field(..., min_length=1, description="List of card IDs to archive")


class BulkArchiveCardsResponse(BaseModel):
    """Schema for bulk archive cards response."""
    success: bool = Field(..., description="Whether the operation succeeded")
    archived_count: int = Field(..., description="Number of cards archived")


class BulkUnarchiveCardsResponse(BaseModel):
    """Schema for bulk unarchive cards response."""
    success: bool = Field(..., description="Whether the operation succeeded")
    unarchived_count: int = Field(..., description="Number of cards unarchived")


class BulkDeleteCardsRequest(BaseModel):
    """Schema for bulk delete cards request."""
    card_ids: list[int] = Field(..., min_length=1, description="List of card IDs to delete")


class BulkDeleteCardsResponse(BaseModel):
    """Schema for bulk delete cards response."""
    success: bool = Field(..., description="Whether the operation succeeded")
    deleted_count: int = Field(..., description="Number of cards deleted")


class BulkLabelCardsRequest(BaseModel):
    """Schema for bulk label cards request."""
    card_ids: list[int] = Field(..., min_length=1, description="List of card IDs to update")
    add_label_ids: Optional[list[int]] = Field(None, description="Label IDs to add")
    remove_label_ids: Optional[list[int]] = Field(None, description="Label IDs to remove")

    @field_validator('add_label_ids', 'remove_label_ids')
    @classmethod
    def validate_not_empty_if_provided(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v


class BulkLabelCardsResponse(BaseModel):
    """Schema for bulk label cards response."""
    success: bool = Field(..., description="Whether the operation succeeded")
    updated_count: int = Field(..., description="Number of cards updated")


# =============================================================================
# Card Filtering Schemas (Phase 3)
# =============================================================================

class FilteredCardsResponse(BaseModel):
    """Schema for filtered cards response."""
    cards: list[CardResponse] = Field(..., description="Filtered cards")
    pagination: PaginationInfo = Field(..., description="Pagination info")


# =============================================================================
# Toggle All Checklist Items Schema (Phase 3)
# =============================================================================

class ToggleAllChecklistItemsRequest(BaseModel):
    """Schema for toggle all checklist items request."""
    checked: bool = Field(..., description="True to check all, False to uncheck all")


# =============================================================================
# Enhanced Card Copy Schema (Phase 3)
# =============================================================================

class CardCopyWithChecklistsRequest(BaseModel):
    """Schema for copying a card with checklists."""
    target_list_id: int = Field(..., description="Destination list ID")
    new_client_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Client-generated unique ID for the copy"
    )
    position: Optional[int] = Field(None, ge=0, description="Position in target list")
    new_title: Optional[str] = Field(None, max_length=500, description="Override title for copy")
    copy_checklists: bool = Field(True, description="Whether to copy checklists")
    copy_labels: bool = Field(True, description="Whether to copy labels")


# =============================================================================
# Search Schemas (Phase 4)
# =============================================================================

SEARCH_MODES = Literal["fts", "vector", "hybrid"]


class SearchRequest(BaseModel):
    """Schema for search request body (used for POST)."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    board_id: Optional[int] = Field(None, description="Filter by board ID")
    label_ids: Optional[list[int]] = Field(None, description="Filter by label IDs (cards must have ALL)")
    priority: Optional[str] = Field(None, description="Filter by priority")
    include_archived: bool = Field(False, description="Include archived cards")
    search_mode: SEARCH_MODES = Field("fts", description="Search mode: fts, vector, or hybrid")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Results to skip")
    page: Optional[int] = Field(None, ge=1, description="Legacy page number (1-indexed)")
    per_page: Optional[int] = Field(None, ge=1, le=100, description="Legacy page size")

    @model_validator(mode="after")
    def normalize_legacy_pagination(self) -> "SearchRequest":
        if self.per_page is not None:
            self.limit = self.per_page
        if self.page is not None:
            self.offset = (self.page - 1) * self.limit
        return self


class SearchResultCard(BaseModel):
    """Schema for a card in search results."""
    id: int = Field(..., description="Card ID")
    uuid: str = Field(..., description="Card UUID")
    board_id: int = Field(..., description="Board ID")
    board_name: str = Field(..., description="Board name")
    list_id: int = Field(..., description="List ID")
    list_name: str = Field(..., description="List name")
    title: str = Field(..., description="Card title")
    description: Optional[str] = Field(None, description="Card description")
    priority: Optional[str] = Field(None, description="Card priority")
    due_date: Optional[datetime] = Field(None, description="Due date")
    labels: list[dict[str, Any]] = Field(default_factory=list, description="Card labels")
    created_at: datetime = Field(..., description="When the card was created")
    updated_at: datetime = Field(..., description="When the card was last updated")
    relevance_score: Optional[float] = Field(None, description="Search relevance score (for vector/hybrid)")

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    """Schema for search response."""
    query: str = Field(..., description="The search query")
    search_mode: str = Field(..., description="Search mode used")
    results: list[SearchResultCard] = Field(..., description="Search results")
    pagination: PaginationInfo = Field(..., description="Pagination info")


# =============================================================================
# Card Links Schemas (Phase 5: Content Integration)
# =============================================================================


class CardLinkCreate(BaseModel):
    """Schema for creating a card link."""

    linked_type: str = Field(..., description="Type of linked content ('media' or 'note')")
    linked_id: str = Field(..., description="ID of the linked content")

    @field_validator("linked_type")
    @classmethod
    def validate_linked_type(cls, v: str) -> str:
        if v not in ("media", "note"):
            raise ValueError("linked_type must be 'media' or 'note'")
        return v


class CardLinkResponse(BaseModel):
    """Schema for card link response."""

    id: int = Field(..., description="Link ID")
    card_id: int = Field(..., description="Card ID")
    linked_type: str = Field(..., description="Type of linked content")
    linked_id: str = Field(..., description="ID of linked content")
    created_at: datetime = Field(..., description="When link was created")

    model_config = ConfigDict(from_attributes=True)


class CardLinksListResponse(BaseModel):
    """Schema for list of card links."""

    links: list[CardLinkResponse] = Field(..., description="List of card links")


class CardLinkCountsResponse(BaseModel):
    """Schema for link counts by type."""

    media: int = Field(0, description="Number of media links")
    note: int = Field(0, description="Number of note links")


class BulkCardLinksRequest(BaseModel):
    """Schema for bulk link operations."""

    links: list[CardLinkCreate] = Field(
        ..., description="Links to add/remove", min_length=1, max_length=100
    )


class BulkCardLinksAddResponse(BaseModel):
    """Schema for bulk add response."""

    added_count: int = Field(..., description="Number of links added")
    skipped_count: int = Field(..., description="Number of duplicates skipped")
    links: list[CardLinkResponse] = Field(..., description="Added links")


class BulkCardLinksRemoveResponse(BaseModel):
    """Schema for bulk remove response."""

    removed_count: int = Field(..., description="Number of links removed")


class LinkedCardResponse(BaseModel):
    """Schema for a card returned in bidirectional lookup."""

    id: int = Field(..., description="Card ID")
    title: str = Field(..., description="Card title")
    description: Optional[str] = Field(None, description="Card description")
    board_id: int = Field(..., description="Board ID")
    board_name: str = Field(..., description="Board name")
    list_id: int = Field(..., description="List ID")
    list_name: str = Field(..., description="List name")
    position: int = Field(..., description="Position in list")
    is_archived: bool = Field(False, description="Whether card is archived")
    is_deleted: bool = Field(False, description="Whether card is soft-deleted")
    link_id: int = Field(..., description="The link ID")
    linked_at: datetime = Field(..., description="When the link was created")

    model_config = ConfigDict(from_attributes=True)


class LinkedCardsListResponse(BaseModel):
    """Schema for bidirectional lookup response."""

    linked_type: str = Field(..., description="Type queried")
    linked_id: str = Field(..., description="Content ID queried")
    cards: list[LinkedCardResponse] = Field(..., description="Cards linked to this content")


# =============================================================================
# Workflow Control Schemas (Safe Orchestrator)
# =============================================================================


class WorkflowPolicyStatusConfig(BaseModel):
    """Status configuration for workflow policy upsert."""

    status_key: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=255)
    is_terminal: bool = Field(False)
    sort_order: int = Field(0)
    is_active: bool = Field(True)


class WorkflowPolicyTransitionConfig(BaseModel):
    """Transition edge configuration for workflow policy upsert."""

    from_status_key: str = Field(..., min_length=1, max_length=128)
    to_status_key: str = Field(..., min_length=1, max_length=128)
    requires_claim: bool = Field(True)
    requires_approval: bool = Field(False)
    approve_to_status_key: Optional[str] = Field(None, min_length=1, max_length=128)
    reject_to_status_key: Optional[str] = Field(None, min_length=1, max_length=128)
    auto_move_list_id: Optional[int] = Field(None)
    max_retries: int = Field(0, ge=0)
    is_active: bool = Field(True)


class WorkflowPolicyUpsertRequest(BaseModel):
    """Workflow policy upsert payload."""

    statuses: Optional[list[WorkflowPolicyStatusConfig]] = None
    transitions: Optional[list[WorkflowPolicyTransitionConfig]] = None
    is_paused: bool = Field(False)
    is_draining: bool = Field(False)
    default_lease_ttl_sec: int = Field(900, gt=0)
    strict_projection: bool = Field(True)
    metadata: Optional[dict[str, Any]] = None


class WorkflowPolicyStatusResponse(BaseModel):
    """Workflow status response item."""

    id: int
    policy_id: int
    status_key: str
    display_name: str
    is_terminal: bool
    sort_order: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class WorkflowPolicyTransitionResponse(BaseModel):
    """Workflow transition response item."""

    id: int
    policy_id: int
    from_status_key: str
    to_status_key: str
    requires_claim: bool
    requires_approval: bool
    approve_to_status_key: Optional[str] = None
    reject_to_status_key: Optional[str] = None
    auto_move_list_id: Optional[int] = None
    max_retries: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class WorkflowPolicyResponse(BaseModel):
    """Workflow policy response payload."""

    id: int
    board_id: int
    version: int
    is_paused: bool
    is_draining: bool
    default_lease_ttl_sec: int
    strict_projection: bool
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    statuses: list[WorkflowPolicyStatusResponse]
    transitions: list[WorkflowPolicyTransitionResponse]


class WorkflowStatusesListResponse(BaseModel):
    """Workflow statuses list response."""

    statuses: list[WorkflowPolicyStatusResponse]


class WorkflowTransitionsListResponse(BaseModel):
    """Workflow transitions list response."""

    transitions: list[WorkflowPolicyTransitionResponse]


class WorkflowStateResponse(BaseModel):
    """Workflow runtime state response."""

    card_id: int
    policy_id: int
    workflow_status_key: str
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    approval_state: str
    pending_transition_id: Optional[int] = None
    retry_counters: Optional[dict[str, Any]] = None
    last_transition_at: Optional[datetime] = None
    last_actor: Optional[str] = None
    version: int
    created_at: datetime
    updated_at: datetime


class WorkflowStatePatchRequest(BaseModel):
    """Patch workflow runtime state payload."""

    workflow_status_key: Optional[str] = Field(None, min_length=1, max_length=128)
    expected_version: int = Field(..., ge=1)
    lease_owner: Optional[str] = None
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    correlation_id: Optional[str] = Field(None, min_length=1, max_length=255)
    actor: Optional[str] = Field(None, min_length=1, max_length=255)


class WorkflowClaimRequest(BaseModel):
    """Claim/lease request payload."""

    owner: str = Field(..., min_length=1, max_length=255)
    lease_ttl_sec: Optional[int] = Field(None, gt=0)
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    correlation_id: Optional[str] = Field(None, min_length=1, max_length=255)


class WorkflowReleaseRequest(BaseModel):
    """Release claim payload."""

    owner: str = Field(..., min_length=1, max_length=255)
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    correlation_id: Optional[str] = Field(None, min_length=1, max_length=255)


class WorkflowTransitionRequest(BaseModel):
    """Workflow transition payload."""

    to_status_key: str = Field(..., min_length=1, max_length=128)
    actor: str = Field(..., min_length=1, max_length=255)
    expected_version: int = Field(..., ge=1)
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    reason: Optional[str] = Field(None, max_length=2000)
    correlation_id: str = Field(..., min_length=1, max_length=255)


class WorkflowApprovalDecisionRequest(BaseModel):
    """Approval decision payload."""

    reviewer: str = Field(..., min_length=1, max_length=255)
    decision: Literal["approved", "rejected"]
    expected_version: int = Field(..., ge=1)
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    reason: Optional[str] = Field(None, max_length=2000)
    correlation_id: str = Field(..., min_length=1, max_length=255)


class WorkflowForceReassignRequest(BaseModel):
    """Force-reassign payload."""

    new_owner: str = Field(..., min_length=1, max_length=255)
    idempotency_key: str = Field(..., min_length=1, max_length=255)
    reason: Optional[str] = Field(None, max_length=2000)
    correlation_id: str = Field(..., min_length=1, max_length=255)


class WorkflowControlResponse(BaseModel):
    """Simple success response for workflow controls."""

    success: bool = Field(True)
    policy: WorkflowPolicyResponse


class WorkflowEventResponse(BaseModel):
    """Workflow event response item."""

    id: int
    card_id: int
    event_type: str
    from_status_key: Optional[str] = None
    to_status_key: Optional[str] = None
    actor: str
    reason: Optional[str] = None
    idempotency_key: str
    correlation_id: Optional[str] = None
    before_snapshot: Optional[dict[str, Any]] = None
    after_snapshot: Optional[dict[str, Any]] = None
    created_at: datetime


class WorkflowEventsListResponse(BaseModel):
    """Workflow events list response."""

    events: list[WorkflowEventResponse]
    pagination: OffsetPaginationMeta


class WorkflowStaleClaimResponse(BaseModel):
    """Stale claim item response."""

    card_id: int
    board_id: int
    list_id: int
    title: str
    workflow_status_key: str
    lease_owner: str
    lease_expires_at: datetime
    version: int
    updated_at: datetime


class WorkflowStaleClaimsListResponse(BaseModel):
    """Stale claims list response."""

    stale_claims: list[WorkflowStaleClaimResponse]
    pagination: OffsetPaginationMeta
