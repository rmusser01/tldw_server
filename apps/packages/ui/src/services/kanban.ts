/**
 * Kanban service - API client for tldw_server kanban endpoints
 */

import { bgRequest, bgUpload } from "./background-proxy"
import type {
  Board,
  BoardCreate,
  BoardUpdate,
  BoardListResponse,
  BoardWithLists,
  KanbanList,
  ListCreate,
  ListUpdate,
  ListsListResponse,
  Card,
  CardCreate,
  CardUpdate,
  CardsListResponse,
  CardMoveRequest,
  ReorderRequest,
  ReorderResponse,
  BoardImportResponse,
  Label,
  LabelCreate,
  LabelUpdate,
  Checklist,
  ChecklistItem,
  ChecklistWithItems,
  Comment,
  PaginationInfo
} from "@/types/kanban"

// Re-export types that consumers import from this module
export type { Checklist, ChecklistItem, ChecklistWithItems, Comment }

// =============================================================================
// Board API
// =============================================================================

/**
 * List all boards
 */
export async function listBoards(params?: {
  limit?: number
  offset?: number
  includeArchived?: boolean
  includeDeleted?: boolean
}): Promise<BoardListResponse> {
  const query = new URLSearchParams()
  if (params?.limit != null) query.set("limit", String(params.limit))
  if (params?.offset != null) query.set("offset", String(params.offset))
  if (params?.includeArchived) query.set("include_archived", "true")
  if (params?.includeDeleted) query.set("include_deleted", "true")
  const suffix = query.toString() ? `?${query.toString()}` : ""
  return await bgRequest<BoardListResponse>({
    path: `/api/v1/kanban/boards${suffix}`,
    method: "GET"
  })
}

/**
 * Get a single board with nested lists and cards
 */
export async function getBoard(
  boardId: number,
  params?: { includeArchived?: boolean }
): Promise<BoardWithLists> {
  const query = new URLSearchParams()
  if (params?.includeArchived) query.set("include_archived", "true")
  const suffix = query.toString() ? `?${query.toString()}` : ""
  return await bgRequest<BoardWithLists>({
    path: `/api/v1/kanban/boards/${boardId}${suffix}`,
    method: "GET"
  })
}

/**
 * Create a new board
 */
export async function createBoard(data: BoardCreate): Promise<Board> {
  return await bgRequest<Board>({
    path: "/api/v1/kanban/boards",
    method: "POST",
    body: data
  })
}

/**
 * Update a board
 */
export async function updateBoard(
  boardId: number,
  data: BoardUpdate,
  expectedVersion?: number
): Promise<Board> {
  const headers: Record<string, string> = {}
  if (expectedVersion != null) {
    headers["X-Expected-Version"] = String(expectedVersion)
  }
  return await bgRequest<Board>({
    path: `/api/v1/kanban/boards/${boardId}`,
    method: "PATCH",
    body: data,
    headers
  })
}

/**
 * Delete a board (soft delete)
 */
export async function deleteBoard(boardId: number): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/boards/${boardId}`,
    method: "DELETE"
  })
}

/**
 * Import a board from JSON (Trello or tldw format)
 */
export async function importBoard(
  data: Record<string, any>,
  boardName?: string
): Promise<BoardImportResponse> {
  return await bgRequest<BoardImportResponse>({
    path: "/api/v1/kanban/boards/import",
    method: "POST",
    body: { data, board_name: boardName }
  })
}

// =============================================================================
// List API
// =============================================================================

/**
 * Get all lists in a board
 */
export async function getLists(boardId: number): Promise<ListsListResponse> {
  return await bgRequest<ListsListResponse>({
    path: `/api/v1/kanban/boards/${boardId}/lists`,
    method: "GET"
  })
}

/**
 * Create a new list
 */
export async function createList(
  boardId: number,
  data: ListCreate
): Promise<KanbanList> {
  return await bgRequest<KanbanList>({
    path: `/api/v1/kanban/boards/${boardId}/lists`,
    method: "POST",
    body: data
  })
}

/**
 * Update a list
 */
export async function updateList(
  listId: number,
  data: ListUpdate,
  expectedVersion?: number
): Promise<KanbanList> {
  const headers: Record<string, string> = {}
  if (expectedVersion != null) {
    headers["X-Expected-Version"] = String(expectedVersion)
  }
  return await bgRequest<KanbanList>({
    path: `/api/v1/kanban/lists/${listId}`,
    method: "PATCH",
    body: data,
    headers
  })
}

/**
 * Delete a list (soft delete)
 */
export async function deleteList(listId: number): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/lists/${listId}`,
    method: "DELETE"
  })
}

/**
 * Reorder lists in a board
 */
export async function reorderLists(
  boardId: number,
  listIds: number[]
): Promise<ReorderResponse> {
  return await bgRequest<ReorderResponse>({
    path: `/api/v1/kanban/boards/${boardId}/lists/reorder`,
    method: "POST",
    body: { ids: listIds } as ReorderRequest
  })
}

// =============================================================================
// Card API
// =============================================================================

/**
 * Get all cards in a list
 */
export async function getCards(listId: number): Promise<CardsListResponse> {
  return await bgRequest<CardsListResponse>({
    path: `/api/v1/kanban/lists/${listId}/cards`,
    method: "GET"
  })
}

/**
 * Get a single card
 */
export async function getCard(cardId: number): Promise<Card> {
  return await bgRequest<Card>({
    path: `/api/v1/kanban/cards/${cardId}`,
    method: "GET"
  })
}

/**
 * Create a new card
 */
export async function createCard(
  listId: number,
  data: CardCreate
): Promise<Card> {
  return await bgRequest<Card>({
    path: `/api/v1/kanban/lists/${listId}/cards`,
    method: "POST",
    body: data
  })
}

/**
 * Update a card
 */
export async function updateCard(
  cardId: number,
  data: CardUpdate,
  expectedVersion?: number
): Promise<Card> {
  const headers: Record<string, string> = {}
  if (expectedVersion != null) {
    headers["X-Expected-Version"] = String(expectedVersion)
  }
  return await bgRequest<Card>({
    path: `/api/v1/kanban/cards/${cardId}`,
    method: "PATCH",
    body: data,
    headers
  })
}

/**
 * Delete a card (soft delete)
 */
export async function deleteCard(cardId: number): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/cards/${cardId}`,
    method: "DELETE"
  })
}

/**
 * Move a card to a different list
 */
export async function moveCard(
  cardId: number,
  targetListId: number,
  position?: number
): Promise<Card> {
  const body: CardMoveRequest = { target_list_id: targetListId }
  if (position != null) body.position = position
  return await bgRequest<Card>({
    path: `/api/v1/kanban/cards/${cardId}/move`,
    method: "POST",
    body
  })
}

/**
 * Reorder cards in a list
 */
export async function reorderCards(
  listId: number,
  cardIds: number[]
): Promise<ReorderResponse> {
  return await bgRequest<ReorderResponse>({
    path: `/api/v1/kanban/lists/${listId}/cards/reorder`,
    method: "POST",
    body: { ids: cardIds } as ReorderRequest
  })
}

// =============================================================================
// Archive / Restore API
// =============================================================================

export async function archiveBoard(boardId: number): Promise<Board> {
  return await bgRequest<Board>({
    path: `/api/v1/kanban/boards/${boardId}/archive`,
    method: "POST"
  })
}

export async function unarchiveBoard(boardId: number): Promise<Board> {
  return await bgRequest<Board>({
    path: `/api/v1/kanban/boards/${boardId}/unarchive`,
    method: "POST"
  })
}

export async function archiveList(listId: number): Promise<KanbanList> {
  return await bgRequest<KanbanList>({
    path: `/api/v1/kanban/lists/${listId}/archive`,
    method: "POST"
  })
}

export async function unarchiveList(listId: number): Promise<KanbanList> {
  return await bgRequest<KanbanList>({
    path: `/api/v1/kanban/lists/${listId}/unarchive`,
    method: "POST"
  })
}

export async function archiveCard(cardId: number): Promise<Card> {
  return await bgRequest<Card>({
    path: `/api/v1/kanban/cards/${cardId}/archive`,
    method: "POST"
  })
}

export async function unarchiveCard(cardId: number): Promise<Card> {
  return await bgRequest<Card>({
    path: `/api/v1/kanban/cards/${cardId}/unarchive`,
    method: "POST"
  })
}

// =============================================================================
// Export API
// =============================================================================

export async function exportBoard(
  boardId: number
): Promise<Record<string, any>> {
  return await bgRequest<Record<string, any>>({
    path: `/api/v1/kanban/boards/${boardId}/export`,
    method: "GET"
  })
}

// =============================================================================
// Label API
// =============================================================================

export async function listLabels(boardId: number): Promise<Label[]> {
  return await bgRequest<Label[]>({
    path: `/api/v1/kanban/boards/${boardId}/labels`,
    method: "GET"
  })
}

export async function createLabel(
  boardId: number,
  data: LabelCreate
): Promise<Label> {
  return await bgRequest<Label>({
    path: `/api/v1/kanban/boards/${boardId}/labels`,
    method: "POST",
    body: data
  })
}

export async function updateLabel(
  labelId: number,
  data: LabelUpdate
): Promise<Label> {
  return await bgRequest<Label>({
    path: `/api/v1/kanban/labels/${labelId}`,
    method: "PATCH",
    body: data
  })
}

export async function deleteLabel(labelId: number): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/labels/${labelId}`,
    method: "DELETE"
  })
}

export async function assignLabelToCard(
  cardId: number,
  labelId: number
): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/cards/${cardId}/labels/${labelId}`,
    method: "POST"
  })
}

export async function removeLabelFromCard(
  cardId: number,
  labelId: number
): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/cards/${cardId}/labels/${labelId}`,
    method: "DELETE"
  })
}

// =============================================================================
// Card Copy API
// =============================================================================

export async function copyCard(cardId: number): Promise<Card> {
  return await bgRequest<Card>({
    path: `/api/v1/kanban/cards/${cardId}/copy`,
    method: "POST"
  })
}

// =============================================================================
// Search API
// =============================================================================

export async function searchCards(params: {
  query: string
  boardId?: number
  limit?: number
}): Promise<CardsListResponse> {
  const qs = new URLSearchParams()
  qs.set("q", params.query)
  if (params.boardId != null) qs.set("board_id", String(params.boardId))
  if (params.limit != null) qs.set("limit", String(params.limit))
  return await bgRequest<CardsListResponse>({
    path: `/api/v1/kanban/cards/search?${qs.toString()}`,
    method: "GET"
  })
}

// =============================================================================
// Checklist API
// =============================================================================

// The API uses name for checklist/item text; components use title/content.
type ApiChecklist = Omit<Checklist, "title"> & { name: string }
type ApiChecklistItem = Omit<ChecklistItem, "content"> & { name: string }
type ApiChecklistWithItems = ApiChecklist & { items: ApiChecklistItem[] }

function toChecklist({ name, ...checklist }: ApiChecklist): Checklist {
  return { ...checklist, title: name }
}

function toChecklistItem({ name, ...item }: ApiChecklistItem): ChecklistItem {
  return { ...item, content: name }
}

export async function listChecklists(
  cardId: number
): Promise<ChecklistWithItems[]> {
  const response = await bgRequest<{ checklists: ApiChecklist[] }>({
    path: `/api/v1/kanban/cards/${cardId}/checklists`,
    method: "GET"
  })
  // The list endpoint returns metadata only. Detail responses include the items
  // needed for rendering each checklist and calculating its progress.
  return await Promise.all(
    response.checklists.map(async (checklist) => {
      const detail = await bgRequest<ApiChecklistWithItems>({
        path: `/api/v1/kanban/checklists/${checklist.id}`,
        method: "GET"
      })
      return {
        ...toChecklist(detail),
        items: detail.items.map(toChecklistItem)
      }
    })
  )
}

export async function createChecklist(
  cardId: number,
  data: { title: string; client_id: string }
): Promise<Checklist> {
  const { title, ...rest } = data
  const response = await bgRequest<ApiChecklist>({
    path: `/api/v1/kanban/cards/${cardId}/checklists`,
    method: "POST",
    body: { ...rest, name: title }
  })
  return toChecklist(response)
}

export async function updateChecklist(
  checklistId: number,
  data: { title?: string }
): Promise<Checklist> {
  const { title, ...rest } = data
  const response = await bgRequest<ApiChecklist>({
    path: `/api/v1/kanban/checklists/${checklistId}`,
    method: "PATCH",
    body: { ...rest, ...(title !== undefined ? { name: title } : {}) }
  })
  return toChecklist(response)
}

export async function deleteChecklist(checklistId: number): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/checklists/${checklistId}`,
    method: "DELETE"
  })
}

export async function createChecklistItem(
  checklistId: number,
  data: { content: string; client_id: string }
): Promise<ChecklistItem> {
  const { content, ...rest } = data
  const response = await bgRequest<ApiChecklistItem>({
    path: `/api/v1/kanban/checklists/${checklistId}/items`,
    method: "POST",
    body: { ...rest, name: content }
  })
  return toChecklistItem(response)
}

export async function updateChecklistItem(
  itemId: number,
  data: { content?: string; checked?: boolean }
): Promise<ChecklistItem> {
  const { content, ...rest } = data
  const response = await bgRequest<ApiChecklistItem>({
    path: `/api/v1/kanban/checklist-items/${itemId}`,
    method: "PATCH",
    body: { ...rest, ...(content !== undefined ? { name: content } : {}) }
  })
  return toChecklistItem(response)
}

export async function deleteChecklistItem(itemId: number): Promise<void> {
  await bgRequest<void>({
    path: `/api/v1/kanban/checklist-items/${itemId}`,
    method: "DELETE"
  })
}

// =============================================================================
// Comment API
// =============================================================================

export async function listComments(cardId: number): Promise<Comment[]> {
  const comments: Comment[] = []
  let offset = 0

  while (true) {
    const response = await bgRequest<{
      comments: Comment[]
      pagination: PaginationInfo
    }>({
      path: `/api/v1/kanban/cards/${cardId}/comments?limit=100&offset=${offset}`,
      method: "GET"
    })
    comments.push(...response.comments)
    if (!response.pagination.has_more) return comments
    if (
      response.comments.length === 0 ||
      response.pagination.offset !== offset
    ) {
      throw new Error("Comment pagination did not advance")
    }
    offset += response.comments.length
  }
}

export async function createComment(
  cardId: number,
  data: { content: string; client_id: string }
): Promise<Comment> {
  return await bgRequest<Comment>({
    path: `/api/v1/kanban/cards/${cardId}/comments`,
    method: "POST",
    body: data
  })
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Generate a unique client ID for idempotency
 */
export function generateClientId(): string {
  return crypto.randomUUID()
}

/**
 * Check if a card is overdue
 */
export function isCardOverdue(card: Card): boolean {
  if (!card.due_date || card.due_complete) return false
  return new Date(card.due_date) < new Date()
}

/**
 * Get priority color
 */
export function getPriorityColor(priority: Card["priority"]): string {
  switch (priority) {
    case "urgent":
      return "#ef4444" // red
    case "high":
      return "#f97316" // orange
    case "medium":
      return "#eab308" // yellow
    case "low":
      return "#3b82f6" // blue
    default:
      return "#9ca3af" // gray
  }
}

/**
 * Format due date for display
 */
export function formatDueDate(dueDate: string | null | undefined): string {
  if (!dueDate) return ""
  const date = new Date(dueDate)
  const now = new Date()
  const diffDays = Math.ceil(
    (date.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
  )

  if (diffDays < 0) {
    return `${Math.abs(diffDays)}d overdue`
  } else if (diffDays === 0) {
    return "Today"
  } else if (diffDays === 1) {
    return "Tomorrow"
  } else if (diffDays < 7) {
    return `${diffDays}d`
  } else {
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric"
    })
  }
}
