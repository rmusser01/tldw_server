import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Drawer, Button, Popconfirm, Empty, Spin, Collapse, message } from "antd"
import { Archive, RotateCcw, Trash2 } from "lucide-react"

import {
  listBoards,
  getBoard,
  unarchiveBoard,
  deleteBoard,
  unarchiveList,
  deleteList,
  unarchiveCard,
  deleteCard
} from "@/services/kanban"
import type { Board, KanbanList, Card, BoardWithLists } from "@/types/kanban"

interface ArchivedItemsDrawerProps {
  open: boolean
  onClose: () => void
  boardId?: number | null
}

export const ArchivedItemsDrawer = ({
  open,
  onClose,
  boardId
}: ArchivedItemsDrawerProps) => {
  const queryClient = useQueryClient()

  // Fetch all boards including archived
  const { data: boardsData, isLoading: boardsLoading } = useQuery({
    queryKey: ["kanban-boards-archived"],
    queryFn: () => listBoards({ limit: 200, includeArchived: true }),
    enabled: open,
    staleTime: 15 * 1000
  })

  // Fetch board data WITH archived items included
  const { data: boardWithArchived, isLoading: boardArchiveLoading } = useQuery({
    queryKey: ["kanban-board-archived", boardId],
    queryFn: () => getBoard(boardId!, { includeArchived: true }),
    enabled: open && boardId != null,
    staleTime: 15 * 1000
  })

  const archivedBoards = (boardsData?.boards ?? []).filter((b) => b.archived)

  // Collect archived lists and cards from board data fetched with include_archived
  const archivedLists: KanbanList[] = []
  const archivedCards: Card[] = []
  if (boardWithArchived) {
    for (const list of boardWithArchived.lists) {
      if (list.archived) archivedLists.push(list)
      for (const card of list.cards) {
        if (card.archived) archivedCards.push(card)
      }
    }
  }

  // Mutations
  const restoreBoardMutation = useMutation({
    mutationFn: (boardId: number) => unarchiveBoard(boardId),
    onSuccess: () => {
      message.success("Board restored")
      queryClient.invalidateQueries({ queryKey: ["kanban-boards"] })
      queryClient.invalidateQueries({ queryKey: ["kanban-boards-archived"] })
    },
    onError: () => {
      message.error("Failed to restore. Please try again.")
    }
  })

  const deleteBoardMutation = useMutation({
    mutationFn: (boardId: number) => deleteBoard(boardId),
    onSuccess: () => {
      message.success("Board deleted")
      queryClient.invalidateQueries({ queryKey: ["kanban-boards"] })
      queryClient.invalidateQueries({ queryKey: ["kanban-boards-archived"] })
    },
    onError: () => {
      message.error("Failed to delete. Please try again.")
    }
  })

  const invalidateBoardQueries = () => {
    if (boardId != null) {
      queryClient.invalidateQueries({ queryKey: ["kanban-board", boardId] })
      queryClient.invalidateQueries({ queryKey: ["kanban-board-archived", boardId] })
    }
  }

  const restoreListMutation = useMutation({
    mutationFn: (listId: number) => unarchiveList(listId),
    onSuccess: () => {
      message.success("List restored")
      invalidateBoardQueries()
    },
    onError: () => {
      message.error("Failed to restore. Please try again.")
    }
  })

  const deleteListMutation = useMutation({
    mutationFn: (listId: number) => deleteList(listId),
    onSuccess: () => {
      message.success("List deleted")
      invalidateBoardQueries()
    },
    onError: () => {
      message.error("Failed to delete. Please try again.")
    }
  })

  const restoreCardMutation = useMutation({
    mutationFn: (cardId: number) => unarchiveCard(cardId),
    onSuccess: () => {
      message.success("Card restored")
      invalidateBoardQueries()
    },
    onError: () => {
      message.error("Failed to restore. Please try again.")
    }
  })

  const deleteCardMutation = useMutation({
    mutationFn: (cardId: number) => deleteCard(cardId),
    onSuccess: () => {
      message.success("Card deleted")
      invalidateBoardQueries()
    },
    onError: () => {
      message.error("Failed to delete. Please try again.")
    }
  })

  const renderItemActions = (
    onRestore: () => void,
    onDelete: () => void,
    name: string
  ) => (
    <div className="flex gap-1">
      <Button
        type="text"
        size="small"
        icon={<RotateCcw className="w-3.5 h-3.5" />}
        onClick={(e) => {
          e.stopPropagation()
          onRestore()
        }}
      >
        Restore
      </Button>
      <Popconfirm
        title={`Delete "${name}"?`}
        description="This cannot be undone."
        onConfirm={onDelete}
        okText="Delete"
        okType="danger"
      >
        <Button
          type="text"
          size="small"
          danger
          icon={<Trash2 className="w-3.5 h-3.5" />}
          onClick={(e) => e.stopPropagation()}
        />
      </Popconfirm>
    </div>
  )

  const collapseItems = [
    ...(archivedBoards.length > 0
      ? [
          {
            key: "boards",
            label: `Archived Boards (${archivedBoards.length})`,
            children: (
              <div className="space-y-2">
                {archivedBoards.map((b) => (
                  <div
                    key={b.id}
                    className="flex items-center justify-between p-2 rounded bg-surface"
                  >
                    <span className="text-sm truncate">{b.name}</span>
                    {renderItemActions(
                      () => restoreBoardMutation.mutate(b.id),
                      () => deleteBoardMutation.mutate(b.id),
                      b.name
                    )}
                  </div>
                ))}
              </div>
            )
          }
        ]
      : []),
    ...(archivedLists.length > 0
      ? [
          {
            key: "lists",
            label: `Archived Lists (${archivedLists.length})`,
            children: (
              <div className="space-y-2">
                {archivedLists.map((l) => (
                  <div
                    key={l.id}
                    className="flex items-center justify-between p-2 rounded bg-surface"
                  >
                    <span className="text-sm truncate">{l.name}</span>
                    {renderItemActions(
                      () => restoreListMutation.mutate(l.id),
                      () => deleteListMutation.mutate(l.id),
                      l.name
                    )}
                  </div>
                ))}
              </div>
            )
          }
        ]
      : []),
    ...(archivedCards.length > 0
      ? [
          {
            key: "cards",
            label: `Archived Cards (${archivedCards.length})`,
            children: (
              <div className="space-y-2">
                {archivedCards.map((c) => (
                  <div
                    key={c.id}
                    className="flex items-center justify-between p-2 rounded bg-surface"
                  >
                    <span className="text-sm truncate">{c.title}</span>
                    {renderItemActions(
                      () => restoreCardMutation.mutate(c.id),
                      () => deleteCardMutation.mutate(c.id),
                      c.title
                    )}
                  </div>
                ))}
              </div>
            )
          }
        ]
      : [])
  ]

  const isEmpty =
    archivedBoards.length === 0 &&
    archivedLists.length === 0 &&
    archivedCards.length === 0

  return (
    <Drawer
      title={
        <div className="flex items-center gap-2">
          <Archive className="w-4 h-4" />
          <span>Archived Items</span>
        </div>
      }
      open={open}
      onClose={onClose}
      size={400}
    >
      {boardsLoading || boardArchiveLoading ? (
        <div className="flex justify-center py-10">
          <Spin />
        </div>
      ) : isEmpty ? (
        <Empty description="No archived items" />
      ) : (
        <Collapse
          items={collapseItems}
          defaultActiveKey={collapseItems.map((i) => i.key)}
          ghost
        />
      )}
    </Drawer>
  )
}
