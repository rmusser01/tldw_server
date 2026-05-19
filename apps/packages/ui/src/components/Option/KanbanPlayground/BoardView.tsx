import { useState, useCallback, useEffect, useRef, useMemo } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Button, Input, message, Popconfirm, Dropdown, Modal, Select } from "antd"
import type { MenuProps } from "antd"
import {
  Plus,
  MoreHorizontal,
  Trash2,
  Calendar,
  GripVertical,
  Edit2,
  CheckSquare,
  MessageCircle,
  FileText,
  Archive,
  Zap,
  Search
} from "lucide-react"
import { DragDropProvider, DragOverlay, type DragDropEvents } from "@dnd-kit/react"
import { useSortable } from "@dnd-kit/react/sortable"
import { closestCorners } from "@dnd-kit/collision"
import { arrayMove } from "@dnd-kit/helpers"

import {
  createList,
  createCard,
  updateBoard,
  updateList,
  deleteList,
  updateCard,
  deleteCard,
  moveCard,
  archiveList,
  unarchiveList,
  archiveCard,
  unarchiveCard,
  reorderLists,
  reorderCards,
  generateClientId,
  isCardOverdue,
  getPriorityColor,
  formatDueDate
} from "@/services/kanban"
import type {
  BoardWithLists,
  ListWithCards,
  Card,
  CardUpdate,
  ListUpdate
} from "@/types/kanban"

import { CardDetailPanel } from "./CardDetailPanel"

interface BoardViewProps {
  board: BoardWithLists
  onRefresh: () => void
  onDelete: () => void
  onArchive?: () => void
  onQuickSetup?: () => void
  /** Expose triggers so the parent can start add-card / add-list from keyboard shortcuts */
  shortcutHandlersRef?: React.MutableRefObject<{
    startAddCard: () => void
    startAddList: () => void
  } | null>
}

type DragStartEvent = Parameters<DragDropEvents["dragstart"]>[0]
type DragEndEvent = Parameters<DragDropEvents["dragend"]>[0]

interface UndoAction {
  type: "archive-card" | "archive-list"
  entityId: number
  label: string
}

export const BoardView = ({
  board,
  onRefresh,
  onDelete,
  onArchive,
  onQuickSetup,
  shortcutHandlersRef
}: BoardViewProps) => {
  const queryClient = useQueryClient()

  // Drag state
  const [activeId, setActiveId] = useState<string | null>(null)
  const [activeType, setActiveType] = useState<"list" | "card" | null>(null)

  // Card detail panel state
  const [selectedCard, setSelectedCard] = useState<Card | null>(null)
  const [detailPanelOpen, setDetailPanelOpen] = useState(false)

  // Board rename state
  const [renameModalOpen, setRenameModalOpen] = useState(false)
  const [renameValue, setRenameValue] = useState(board.name)

  // Filter state
  const [filterLabelIds, setFilterLabelIds] = useState<number[]>([])
  const [filterPriorities, setFilterPriorities] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const hasFilters = filterLabelIds.length > 0 || filterPriorities.length > 0 || searchQuery.length > 0

  // Add list state
  const [addingList, setAddingList] = useState(false)
  const [newListName, setNewListName] = useState("")

  // Add card state - tracks which list is in "add card" mode
  const [addingCardListId, setAddingCardListId] = useState<number | null>(null)

  // Expose add-card / add-list triggers to parent via ref
  useEffect(() => {
    if (shortcutHandlersRef) {
      shortcutHandlersRef.current = {
        startAddCard: () => {
          // Open add-card on the first list
          const firstList = board.lists[0]
          if (firstList) setAddingCardListId(firstList.id)
        },
        startAddList: () => {
          setAddingList(true)
        }
      }
    }
    return () => {
      if (shortcutHandlersRef) shortcutHandlersRef.current = null
    }
  }, [shortcutHandlersRef, board.lists])

  // Undo state
  const undoRef = useRef<UndoAction | null>(null)

  useEffect(() => {
    setRenameValue(board.name)
  }, [board.id, board.name])

  // Show undo toast after archive
  const showUndoToast = useCallback(
    (action: UndoAction) => {
      undoRef.current = action
      message.open({
        type: "success",
        content: (
          <div className="flex items-center gap-2">
            <span>{`"${action.label}" archived`}</span>
            <Button
              type="link"
              size="small"
              onClick={() => {
                const a = undoRef.current
                if (!a) return
                undoRef.current = null
                if (a.type === "archive-card") {
                  unarchiveCard(a.entityId)
                    .then(() => {
                      queryClient.invalidateQueries({
                        queryKey: ["kanban-board", board.id]
                      })
                      message.success("Card restored")
                    })
                    .catch(() => {
                      message.error("Failed to restore card. Please try again.")
                    })
                } else if (a.type === "archive-list") {
                  unarchiveList(a.entityId)
                    .then(() => {
                      queryClient.invalidateQueries({
                        queryKey: ["kanban-board", board.id]
                      })
                      message.success("List restored")
                    })
                    .catch(() => {
                      message.error("Failed to restore list. Please try again.")
                    })
                }
              }}
            >
              Undo
            </Button>
          </div>
        ),
        duration: 10
      })
    },
    [board.id, queryClient]
  )

  // Mutations
  const createListMutation = useMutation({
    mutationFn: (name: string) =>
      createList(board.id, { name, client_id: generateClientId() }),
    onSuccess: () => {
      message.success("List created")
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
      setAddingList(false)
      setNewListName("")
    },
    onError: () => {
      message.error("Failed to create list. Please try again.")
    }
  })

  const updateBoardMutation = useMutation({
    mutationFn: (name: string) => updateBoard(board.id, { name }),
    onSuccess: () => {
      message.success("Board renamed")
      queryClient.invalidateQueries({ queryKey: ["kanban-boards"] })
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
      setRenameModalOpen(false)
    },
    onError: () => {
      message.error("Failed to rename board. Please try again.")
    }
  })

  const updateListMutation = useMutation({
    mutationFn: ({ listId, data }: { listId: number; data: ListUpdate }) =>
      updateList(listId, data),
    onSuccess: () => {
      message.success("List renamed")
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
    },
    onError: () => {
      message.error("Failed to rename list. Please try again.")
    }
  })

  const deleteListMutation = useMutation({
    mutationFn: (listId: number) => deleteList(listId),
    onSuccess: () => {
      message.success("List deleted")
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
    },
    onError: () => {
      message.error("Failed to delete list. Please try again.")
    }
  })

  const archiveListMutation = useMutation({
    mutationFn: ({ listId, listName }: { listId: number; listName: string }) =>
      archiveList(listId),
    onSuccess: (_data, { listId, listName }) => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
      showUndoToast({ type: "archive-list", entityId: listId, label: listName })
    },
    onError: () => {
      message.error("Failed to archive list. Please try again.")
    }
  })

  const archiveCardMutation = useMutation({
    mutationFn: ({
      cardId,
      cardTitle
    }: {
      cardId: number
      cardTitle: string
    }) => archiveCard(cardId),
    onSuccess: (_data, { cardId, cardTitle }) => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
      setDetailPanelOpen(false)
      setSelectedCard(null)
      showUndoToast({
        type: "archive-card",
        entityId: cardId,
        label: cardTitle
      })
    },
    onError: () => {
      message.error("Failed to archive card. Please try again.")
    }
  })

  const createCardMutation = useMutation({
    mutationFn: ({ listId, title }: { listId: number; title: string }) =>
      createCard(listId, { title, client_id: generateClientId() }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
      setAddingCardListId(null)
    },
    onError: () => {
      message.error("Failed to create card. Please try again.")
    }
  })

  const updateCardMutation = useMutation({
    mutationFn: ({ cardId, data }: { cardId: number; data: CardUpdate }) =>
      updateCard(cardId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
    },
    onError: () => {
      message.error("Failed to update card. Please try again.")
    }
  })

  const deleteCardMutation = useMutation({
    mutationFn: (cardId: number) => deleteCard(cardId),
    onSuccess: () => {
      message.success("Card deleted")
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
      setDetailPanelOpen(false)
      setSelectedCard(null)
    },
    onError: () => {
      message.error("Failed to delete card. Please try again.")
    }
  })

  const moveCardMutation = useMutation({
    mutationFn: ({
      cardId,
      targetListId,
      position
    }: {
      cardId: number
      targetListId: number
      position?: number
    }) => moveCard(cardId, targetListId, position),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
    },
    onError: () => {
      message.error("Failed to move card. Please try again.")
    }
  })

  const reorderListsMutation = useMutation({
    mutationFn: (listIds: number[]) => reorderLists(board.id, listIds),
    onMutate: async (listIds) => {
      await queryClient.cancelQueries({ queryKey: ["kanban-board", board.id] })
      const previous = queryClient.getQueryData<BoardWithLists>(["kanban-board", board.id])
      if (previous) {
        const reordered = listIds
          .map((id) => previous.lists.find((l) => l.id === id))
          .filter(Boolean) as ListWithCards[]
        queryClient.setQueryData<BoardWithLists>(["kanban-board", board.id], {
          ...previous,
          lists: reordered
        })
      }
      return { previous }
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(["kanban-board", board.id], context.previous)
      }
      message.error("Failed to reorder lists. Please try again.")
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
    }
  })

  const reorderCardsMutation = useMutation({
    mutationFn: ({ listId, cardIds }: { listId: number; cardIds: number[] }) =>
      reorderCards(listId, cardIds),
    onMutate: async ({ listId, cardIds }) => {
      await queryClient.cancelQueries({ queryKey: ["kanban-board", board.id] })
      const previous = queryClient.getQueryData<BoardWithLists>(["kanban-board", board.id])
      if (previous) {
        const updated = {
          ...previous,
          lists: previous.lists.map((l) => {
            if (l.id !== listId) return l
            const reordered = cardIds
              .map((id) => l.cards.find((c) => c.id === id))
              .filter(Boolean) as Card[]
            return { ...l, cards: reordered }
          })
        }
        queryClient.setQueryData<BoardWithLists>(["kanban-board", board.id], updated)
      }
      return { previous }
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(["kanban-board", board.id], context.previous)
      }
      message.error("Failed to reorder cards. Please try again.")
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", board.id] })
    }
  })

  // Handle adding new list
  const handleAddList = useCallback(() => {
    if (!newListName.trim()) return
    createListMutation.mutate(newListName.trim())
  }, [newListName, createListMutation])

  // Open card detail panel
  const handleCardClick = useCallback((card: Card) => {
    setSelectedCard(card)
    setDetailPanelOpen(true)
  }, [])

  // Save card from detail panel
  const handleSaveCard = useCallback(
    async (cardId: number, data: CardUpdate) => {
      const updatedCard = await updateCardMutation.mutateAsync({ cardId, data })
      setSelectedCard((current) =>
        current?.id === updatedCard.id ? updatedCard : current
      )
    },
    [updateCardMutation]
  )

  // Delete card
  const handleDeleteCard = useCallback(
    (cardId: number) => {
      deleteCardMutation.mutate(cardId)
    },
    [deleteCardMutation]
  )

  // Rename list
  const handleRenameList = useCallback(
    (listId: number, name: string) => {
      updateListMutation.mutate({ listId, data: { name } })
    },
    [updateListMutation]
  )

  // Archive list
  const handleArchiveList = useCallback(
    (listId: number, listName: string) => {
      archiveListMutation.mutate({ listId, listName })
    },
    [archiveListMutation]
  )

  // Archive card
  const handleArchiveCard = useCallback(
    (cardId: number, cardTitle: string) => {
      archiveCardMutation.mutate({ cardId, cardTitle })
    },
    [archiveCardMutation]
  )

  // Drag handlers
  const handleDragStart = useCallback((event: DragStartEvent) => {
    const sourceId = event.operation.source?.id
    if (!sourceId) return
    const idStr = String(sourceId)
    setActiveId(idStr)

    if (idStr.startsWith("list-")) {
      setActiveType("list")
    } else if (idStr.startsWith("card-")) {
      setActiveType("card")
    }
  }, [])

  const handleRenameBoard = useCallback(() => {
    const nextName = renameValue.trim()
    if (!nextName) {
      message.warning("Please enter a board name")
      return
    }
    if (nextName === board.name) {
      setRenameModalOpen(false)
      return
    }
    updateBoardMutation.mutate(nextName)
  }, [renameValue, board.name, updateBoardMutation])

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      if (event.canceled) {
        setActiveId(null)
        setActiveType(null)
        return
      }

      const sourceId = event.operation.source?.id
      const targetId = event.operation.target?.id

      if (!sourceId || !targetId) {
        setActiveId(null)
        setActiveType(null)
        return
      }

      const activeIdStr = String(sourceId)
      const overIdStr = String(targetId)

      // Handle list reordering
      if (activeIdStr.startsWith("list-") && overIdStr.startsWith("list-")) {
        const activeListId = parseInt(activeIdStr.replace("list-", ""))
        const overListId = parseInt(overIdStr.replace("list-", ""))

        if (activeListId !== overListId) {
          const oldIndex = board.lists.findIndex((l) => l.id === activeListId)
          const newIndex = board.lists.findIndex((l) => l.id === overListId)

          if (oldIndex !== -1 && newIndex !== -1) {
            const newOrder = arrayMove(board.lists, oldIndex, newIndex)
            reorderListsMutation.mutate(newOrder.map((l) => l.id))
          }
        }
      }

      // Handle card reordering within same list
      if (activeIdStr.startsWith("card-") && overIdStr.startsWith("card-")) {
        const activeCardId = parseInt(activeIdStr.replace("card-", ""))
        const overCardId = parseInt(overIdStr.replace("card-", ""))

        // Find which list contains the active card
        const sourceList = board.lists.find((l) =>
          l.cards.some((c) => c.id === activeCardId)
        )
        const targetList = board.lists.find((l) =>
          l.cards.some((c) => c.id === overCardId)
        )

        if (sourceList && targetList) {
          if (sourceList.id === targetList.id) {
            // Same list - reorder
            const oldIndex = sourceList.cards.findIndex(
              (c) => c.id === activeCardId
            )
            const newIndex = sourceList.cards.findIndex(
              (c) => c.id === overCardId
            )

            if (oldIndex !== -1 && newIndex !== -1 && oldIndex !== newIndex) {
              const newOrder = arrayMove(sourceList.cards, oldIndex, newIndex)
              reorderCardsMutation.mutate({
                listId: sourceList.id,
                cardIds: newOrder.map((c) => c.id)
              })
            }
          } else {
            // Different list - move card
            const targetIndex = targetList.cards.findIndex(
              (c) => c.id === overCardId
            )
            moveCardMutation.mutate({
              cardId: activeCardId,
              targetListId: targetList.id,
              position: targetIndex >= 0 ? targetIndex : undefined
            })
          }
        }
      }

      // Handle dropping card on a list (move to end of list)
      if (activeIdStr.startsWith("card-") && overIdStr.startsWith("list-")) {
        const activeCardId = parseInt(activeIdStr.replace("card-", ""))
        const targetListId = parseInt(overIdStr.replace("list-", ""))

        const sourceList = board.lists.find((l) =>
          l.cards.some((c) => c.id === activeCardId)
        )

        if (sourceList && sourceList.id !== targetListId) {
          moveCardMutation.mutate({
            cardId: activeCardId,
            targetListId
          })
        }
      }

      setActiveId(null)
      setActiveType(null)
    },
    [
      board.lists,
      reorderListsMutation,
      reorderCardsMutation,
      moveCardMutation
    ]
  )

  // Find active card for drag overlay
  const getActiveCard = (): Card | null => {
    if (!activeId || activeType !== "card") return null
    const cardId = parseInt(activeId.replace("card-", ""))
    for (const list of board.lists) {
      const card = list.cards.find((c) => c.id === cardId)
      if (card) return card
    }
    return null
  }

  // Find active list for drag overlay
  const getActiveList = (): ListWithCards | null => {
    if (!activeId || activeType !== "list") return null
    const listId = parseInt(activeId.replace("list-", ""))
    return board.lists.find((l) => l.id === listId) || null
  }

  // Collect all unique labels from cards for filter options
  const allLabels = useMemo(() => {
    const map = new Map<number, { id: number; name: string; color: string }>()
    for (const list of board.lists) {
      for (const card of list.cards) {
        for (const label of card.labels ?? []) {
          map.set(label.id, { id: label.id, name: label.name, color: label.color })
        }
      }
    }
    return Array.from(map.values())
  }, [board.lists])

  const handleClearFilters = useCallback(() => {
    setFilterLabelIds([])
    setFilterPriorities([])
    setSearchQuery("")
  }, [])

  // Check if card matches current filters
  const cardMatchesFilters = useCallback(
    (card: Card) => {
      if (!hasFilters) return true
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        const matchesTitle = card.title.toLowerCase().includes(q)
        const matchesDesc = (card.description ?? "").toLowerCase().includes(q)
        if (!matchesTitle && !matchesDesc) return false
      }
      if (filterPriorities.length > 0) {
        if (!card.priority || !filterPriorities.includes(card.priority)) {
          return false
        }
      }
      if (filterLabelIds.length > 0) {
        const cardLabelIds = (card.labels ?? []).map((l) => l.id)
        if (!filterLabelIds.some((id) => cardLabelIds.includes(id))) {
          return false
        }
      }
      return true
    },
    [hasFilters, searchQuery, filterPriorities, filterLabelIds]
  )

  return (
    <div className="board-view">
      {/* Board header with name and actions */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-medium">{board.name}</h2>
          <Button
            type="text"
            size="small"
            aria-label="Rename board"
            icon={<Edit2 className="w-4 h-4" />}
            onClick={() => {
              setRenameValue(board.name)
              setRenameModalOpen(true)
            }}
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-text-muted">
            {board.lists.length} lists, {board.total_cards} cards
          </span>
          {onQuickSetup && (
            <Button
              size="small"
              icon={<Zap className="w-4 h-4" />}
              onClick={onQuickSetup}
            >
              Quick Setup
            </Button>
          )}
          {onArchive && (
            <Button
              size="small"
              icon={<Archive className="w-4 h-4" />}
              onClick={onArchive}
            >
              Archive Board
            </Button>
          )}
          <Popconfirm
            title="Delete this board?"
            description="All lists and cards will be removed. This cannot be undone."
            onConfirm={onDelete}
            okText="Delete"
            okType="danger"
          >
            <Button danger type="text" icon={<Trash2 className="w-4 h-4" />} size="small">
              Delete
            </Button>
          </Popconfirm>
        </div>
      </div>

      {/* Filter bar */}
      {(allLabels.length > 0 || board.total_cards > 0) && (
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <Input
            placeholder="Search cards..."
            prefix={<Search className="w-3.5 h-3.5 text-text-muted" />}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            allowClear
            size="small"
            style={{ width: 180 }}
          />
          {allLabels.length > 0 && (
            <Select
              mode="multiple"
              placeholder="Filter by label"
              style={{ minWidth: 160 }}
              value={filterLabelIds}
              onChange={setFilterLabelIds}
              maxTagCount={2}
              allowClear
              size="small"
              options={allLabels.map((l) => ({
                value: l.id,
                label: (
                  <div className="flex items-center gap-1">
                    <span
                      className="inline-block w-3 h-3 rounded-sm"
                      style={{ backgroundColor: l.color }}
                    />
                    {l.name}
                  </div>
                )
              }))}
            />
          )}
          <Select
            mode="multiple"
            placeholder="Filter by priority"
            style={{ minWidth: 150 }}
            value={filterPriorities}
            onChange={setFilterPriorities}
            maxTagCount={2}
            allowClear
            size="small"
            options={[
              { value: "urgent", label: "Urgent" },
              { value: "high", label: "High" },
              { value: "medium", label: "Medium" },
              { value: "low", label: "Low" }
            ]}
          />
          {hasFilters && (
            <Button
              size="small"
              type="link"
              onClick={handleClearFilters}
            >
              Clear filters
            </Button>
          )}
        </div>
      )}

      {/* Kanban board with DnD */}
      <DragDropProvider onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
        <div className="flex gap-4 overflow-x-auto pb-4 min-h-[400px]">
          {board.lists.map((list, index) => (
            <SortableList
              key={list.id}
              list={list}
              index={index}
              onDeleteList={() => deleteListMutation.mutate(list.id)}
              onArchiveList={() =>
                handleArchiveList(list.id, list.name)
              }
              onRenameList={(name) => handleRenameList(list.id, name)}
              onCardClick={handleCardClick}
              addingCard={addingCardListId === list.id}
              onStartAddCard={() => setAddingCardListId(list.id)}
              onCancelAddCard={() => setAddingCardListId(null)}
              onAddCard={(title) =>
                createCardMutation.mutate({ listId: list.id, title })
              }
              createCardLoading={createCardMutation.isPending}
              cardMatchesFilters={cardMatchesFilters}
            />
          ))}

          {/* Add list button/input */}
          <div className="flex-shrink-0 w-72">
            {addingList ? (
              <div className="bg-surface rounded-lg p-3">
                <Input
                  placeholder="Enter list name"
                  value={newListName}
                  onChange={(e) => setNewListName(e.target.value)}
                  onPressEnter={handleAddList}
                  autoFocus
                />
                <div className="flex gap-2 mt-2">
                  <Button
                    type="primary"
                    size="small"
                    onClick={handleAddList}
                    loading={createListMutation.isPending}
                  >
                    Add List
                  </Button>
                  <Button
                    size="small"
                    onClick={() => {
                      setAddingList(false)
                      setNewListName("")
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <Button
                type="dashed"
                className="w-full h-10"
                icon={<Plus className="w-4 h-4" />}
                onClick={() => setAddingList(true)}
              >
                Add List
              </Button>
            )}
          </div>
        </div>

        {/* Drag overlay */}
        <DragOverlay>
          {activeType === "card" && getActiveCard() && (
            <KanbanCardPreview card={getActiveCard()!} isDragging />
          )}
          {activeType === "list" && getActiveList() && (
            <div className="bg-surface rounded-lg p-3 w-72 opacity-80">
              <div className="font-medium">{getActiveList()!.name}</div>
              <div className="text-sm text-text-muted">
                {getActiveList()!.cards.length} cards
              </div>
            </div>
          )}
        </DragOverlay>
      </DragDropProvider>

      {/* Card detail panel */}
      <CardDetailPanel
        card={selectedCard}
        boardId={board.id}
        lists={board.lists}
        open={detailPanelOpen}
        onClose={() => {
          setDetailPanelOpen(false)
          setSelectedCard(null)
        }}
        onSave={handleSaveCard}
        onDelete={handleDeleteCard}
        onArchive={(cardId, cardTitle) =>
          handleArchiveCard(cardId, cardTitle)
        }
        onMove={(cardId, targetListId) =>
          moveCardMutation.mutate({ cardId, targetListId })
        }
        onCopied={() => {
          queryClient.invalidateQueries({
            queryKey: ["kanban-board", board.id]
          })
        }}
      />

      <Modal
        title="Rename Board"
        open={renameModalOpen}
        onCancel={() => setRenameModalOpen(false)}
        onOk={handleRenameBoard}
        okText="Save"
        confirmLoading={updateBoardMutation.isPending}
        okButtonProps={{ disabled: !renameValue.trim() }}
      >
        <div className="py-4">
          <label className="block text-sm font-medium mb-2">Board Name</label>
          <Input
            placeholder="Enter board name"
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onPressEnter={handleRenameBoard}
            autoFocus
          />
        </div>
      </Modal>
    </div>
  )
}

// =============================================================================
// Sortable List Component
// =============================================================================

interface SortableListProps {
  list: ListWithCards
  index: number
  onDeleteList: () => void
  onArchiveList: () => void
  onRenameList: (name: string) => void
  onCardClick: (card: Card) => void
  addingCard: boolean
  onStartAddCard: () => void
  onCancelAddCard: () => void
  onAddCard: (title: string) => void
  createCardLoading: boolean
  cardMatchesFilters: (card: Card) => boolean
}

const SortableList = ({
  list,
  index,
  onDeleteList,
  onArchiveList,
  onRenameList,
  onCardClick,
  addingCard,
  onStartAddCard,
  onCancelAddCard,
  onAddCard,
  createCardLoading,
  cardMatchesFilters
}: SortableListProps) => {
  const {
    ref,
    handleRef,
    isDragging
  } = useSortable({
    id: `list-${list.id}`,
    index,
    collisionDetector: closestCorners,
    group: "lists",
    plugins: []
  })

  // Local state for inline rename
  const [isRenaming, setIsRenaming] = useState(false)
  const [renameValue, setRenameValue] = useState(list.name)

  // Local state for card title (fixes shared newCardTitle bug)
  const [newCardTitle, setNewCardTitle] = useState("")

  // Delete confirmation state
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)

  const style = {
    opacity: isDragging ? 0.5 : 1
  }

  const handleStartRename = () => {
    setRenameValue(list.name)
    setIsRenaming(true)
  }

  const handleConfirmRename = () => {
    const trimmed = renameValue.trim()
    if (trimmed && trimmed !== list.name) {
      onRenameList(trimmed)
    }
    setIsRenaming(false)
  }

  const handleCancelRename = () => {
    setRenameValue(list.name)
    setIsRenaming(false)
  }

  const handleAddCard = () => {
    const trimmed = newCardTitle.trim()
    if (!trimmed) return
    onAddCard(trimmed)
    setNewCardTitle("")
  }

  const handleCancelAddCard = () => {
    setNewCardTitle("")
    onCancelAddCard()
  }

  const menuItems: MenuProps["items"] = [
    {
      key: "rename",
      label: "Rename List",
      icon: <Edit2 className="w-4 h-4" />,
      onClick: handleStartRename
    },
    { type: "divider" },
    {
      key: "archive",
      label: "Archive List",
      icon: <Archive className="w-4 h-4" />,
      onClick: onArchiveList
    },
    {
      key: "delete",
      label: "Delete List",
      danger: true,
      icon: <Trash2 className="w-4 h-4" />,
      onClick: () => setDeleteConfirmOpen(true)
    }
  ]

  return (
    <div
      ref={ref}
      style={style}
      className="kanban-list flex-shrink-0 w-72 bg-surface rounded-lg"
    >
      {/* List header */}
      <div
        className="flex items-center justify-between p-3 cursor-grab"
        ref={handleRef}
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <GripVertical className="w-4 h-4 text-text-subtle flex-shrink-0" />
          {isRenaming ? (
            <Input
              size="small"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onPressEnter={handleConfirmRename}
              onBlur={handleConfirmRename}
              onKeyDown={(e) => {
                if (e.key === "Escape") handleCancelRename()
              }}
              autoFocus
              className="flex-1"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span
              className="font-medium truncate cursor-text"
              onDoubleClick={handleStartRename}
              title={list.name}
            >
              {list.name}
            </span>
          )}
          <span className="text-xs text-text-muted bg-surface2 px-1.5 py-0.5 rounded flex-shrink-0">
            {list.cards.length}
          </span>
        </div>
        <Popconfirm
          title={`Delete list '${list.name}'?`}
          description={`${list.cards.length} card${list.cards.length !== 1 ? "s" : ""} will be removed.`}
          open={deleteConfirmOpen}
          onConfirm={() => {
            setDeleteConfirmOpen(false)
            onDeleteList()
          }}
          onCancel={() => setDeleteConfirmOpen(false)}
          okText="Delete"
          okType="danger"
        >
          <span className="absolute w-0 h-0 overflow-hidden" />
        </Popconfirm>
        <Dropdown menu={{ items: menuItems }} trigger={["click"]}>
          <Button
            type="text"
            size="small"
            icon={<MoreHorizontal className="w-4 h-4" />}
          />
        </Dropdown>
      </div>

      {/* Cards container - dynamic max height */}
      <div className="px-2 pb-2 max-h-[calc(100vh-250px)] overflow-y-auto">
        {list.cards.map((card, cardIndex) => (
          <SortableCard
            key={card.id}
            card={card}
            index={cardIndex}
            group={`cards-${list.id}`}
            onClick={() => onCardClick(card)}
            dimmed={!cardMatchesFilters(card)}
          />
        ))}

        {/* Add card input */}
        {addingCard ? (
          <div className="mt-2">
            <Input.TextArea
              placeholder="Enter card title"
              value={newCardTitle}
              onChange={(e) => setNewCardTitle(e.target.value)}
              autoSize={{ minRows: 2, maxRows: 4 }}
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleAddCard()
                }
                if (e.key === "Escape") {
                  handleCancelAddCard()
                }
              }}
            />
            <div className="flex gap-2 mt-2">
              <Button
                type="primary"
                size="small"
                onClick={handleAddCard}
                loading={createCardLoading}
              >
                Add
              </Button>
              <Button size="small" onClick={handleCancelAddCard}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <Button
            type="text"
            className="w-full mt-2 text-text-muted hover:text-text"
            icon={<Plus className="w-4 h-4" />}
            onClick={onStartAddCard}
          >
            Add card
          </Button>
        )}
      </div>
    </div>
  )
}

// =============================================================================
// Sortable Card Component
// =============================================================================

interface SortableCardProps {
  card: Card
  index: number
  group: string
  onClick: () => void
  dimmed?: boolean
}

const SortableCard = ({ card, index, group, onClick, dimmed }: SortableCardProps) => {
  const {
    ref,
    isDragging
  } = useSortable({
    id: `card-${card.id}`,
    index,
    collisionDetector: closestCorners,
    group,
    type: "card",
    plugins: []
  })

  const style = {
    opacity: isDragging ? 0.5 : dimmed ? 0.3 : 1
  }

  return (
    <div
      ref={ref}
      style={style}
      onClick={onClick}
      className={`kanban-card bg-elevated rounded-md mb-2 shadow-sm cursor-pointer hover:shadow-md transition-shadow overflow-hidden${
        dimmed ? " pointer-events-none" : ""
      }`}
    >
      {/* Priority left border */}
      <div className="flex">
        {card.priority && (
          <div
            className="w-1 flex-shrink-0 rounded-l-md"
            style={{ backgroundColor: getPriorityColor(card.priority) }}
          />
        )}
        <div className="flex-1 p-3 min-w-0">
          <KanbanCardPreview card={card} />
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Card Preview Component (used in both card and drag overlay)
// =============================================================================

interface KanbanCardPreviewProps {
  card: Card
  isDragging?: boolean
}

const KanbanCardPreview = ({ card, isDragging }: KanbanCardPreviewProps) => {
  const overdue = isCardOverdue(card)
  const hasLabels = card.labels && card.labels.length > 0
  const hasChecklist = (card.checklist_total ?? 0) > 0
  const hasComments = (card.comment_count ?? 0) > 0
  const hasDescription = !!card.description

  return (
    <div className={isDragging ? "bg-elevated rounded-md p-3 shadow-lg" : ""}>
      {/* Label color bars */}
      {hasLabels && (
        <div className="flex items-center gap-1 flex-wrap mb-1.5">
          {card.labels!.map((label) => (
            <span
              key={label.id}
              className="inline-block w-8 h-1.5 rounded-sm"
              style={{ backgroundColor: label.color }}
              title={label.name}
            />
          ))}
        </div>
      )}

      {/* Title */}
      <div className="text-sm font-medium mb-1">{card.title}</div>

      {/* Badges row */}
      <div className="flex items-center gap-2 flex-wrap">
        {/* Due date badge */}
        {card.due_date && (
          <span
            className={`text-xs flex items-center gap-1 px-1.5 py-0.5 rounded ${
              overdue
                ? "bg-danger/10 text-danger"
                : card.due_complete
                  ? "bg-success/10 text-success"
                  : "bg-surface text-text-muted"
            }`}
          >
            <Calendar className="w-3 h-3" />
            {formatDueDate(card.due_date)}
          </span>
        )}

        {/* Description indicator */}
        {hasDescription && (
          <span className="text-text-muted" title="Has description">
            <FileText className="w-3.5 h-3.5" />
          </span>
        )}

        {/* Checklist progress */}
        {hasChecklist && (
          <span
            className={`text-xs flex items-center gap-1 ${
              card.checklist_complete === card.checklist_total
                ? "text-success"
                : "text-text-muted"
            }`}
            title="Checklist progress"
          >
            <CheckSquare className="w-3.5 h-3.5" />
            {card.checklist_complete}/{card.checklist_total}
          </span>
        )}

        {/* Comment count */}
        {hasComments && (
          <span
            className="text-xs flex items-center gap-1 text-text-muted"
            title={`${card.comment_count} comment${card.comment_count !== 1 ? "s" : ""}`}
          >
            <MessageCircle className="w-3.5 h-3.5" />
            {card.comment_count}
          </span>
        )}
      </div>
    </div>
  )
}
