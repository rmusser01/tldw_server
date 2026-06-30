import { useState, useEffect } from "react"
import {
  Drawer,
  Input,
  Select,
  Button,
  Popconfirm,
  Space,
  message
} from "antd"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Trash2, Archive, Copy, Send } from "lucide-react"

import type { Card, CardUpdate, ListWithCards, PriorityType, Comment } from "@/types/kanban"
import { copyCard, listComments, createComment, generateClientId } from "@/services/kanban"
import { LabelManager } from "./LabelManager"
import { ChecklistSection } from "./ChecklistSection"
import {
  formatKanbanDateTimeLocalValue,
  parseKanbanDateTimeLocalValue
} from "./kanbanDateTime"

interface CardDetailPanelProps {
  card: Card | null
  boardId: number
  lists: ListWithCards[]
  open: boolean
  onClose: () => void
  onSave: (cardId: number, data: CardUpdate) => void | Promise<void>
  onDelete: (cardId: number) => void
  onArchive?: (cardId: number, cardTitle: string) => void
  onMove: (cardId: number, targetListId: number) => void
  onCopied?: () => void
}

const PRIORITY_OPTIONS = [
  { value: "low", label: "Low", color: "rgb(var(--color-primary))" },
  { value: "medium", label: "Medium", color: "rgb(var(--color-warn))" },
  { value: "high", label: "High", color: "rgb(var(--color-warn))" },
  { value: "urgent", label: "Urgent", color: "rgb(var(--color-danger))" }
]

export const CardDetailPanel = ({
  card,
  boardId,
  lists,
  open,
  onClose,
  onSave,
  onDelete,
  onArchive,
  onMove,
  onCopied
}: CardDetailPanelProps) => {
  const queryClient = useQueryClient()

  // Form state
  const [title, setTitle] = useState("")
  const [description, setDescription] = useState("")
  const [dueDateInput, setDueDateInput] = useState("")
  const [dueDateTouched, setDueDateTouched] = useState(false)
  const [priority, setPriority] = useState<PriorityType | null>(null)
  const [newComment, setNewComment] = useState("")

  // Track if form is dirty
  const [isDirty, setIsDirty] = useState(false)

  // Copy card mutation
  const copyCardMutation = useMutation({
    mutationFn: (cardId: number) => copyCard(cardId),
    onSuccess: () => {
      message.success("Card copied")
      onCopied?.()
    },
    onError: () => {
      message.error("Failed to copy card. Please try again.")
    }
  })

  // Comments query
  const { data: comments = [] } = useQuery({
    queryKey: ["kanban-card-comments", card?.id],
    queryFn: () => listComments(card!.id),
    enabled: !!card?.id && open,
    staleTime: 30 * 1000
  })

  // Add comment mutation
  const addCommentMutation = useMutation({
    mutationFn: ({ cardId, content }: { cardId: number; content: string }) =>
      createComment(cardId, { content, client_id: generateClientId() }),
    onSuccess: () => {
      setNewComment("")
      queryClient.invalidateQueries({
        queryKey: ["kanban-card-comments", card?.id]
      })
    },
    onError: () => {
      message.error("Failed to add comment. Please try again.")
    }
  })

  // Sync form state when card changes
  useEffect(() => {
    if (card) {
      setTitle(card.title)
      setDescription(card.description || "")
      setDueDateInput(formatKanbanDateTimeLocalValue(card.due_date))
      setDueDateTouched(false)
      setPriority(card.priority || null)
      setIsDirty(false)
    }
  }, [card])

  const handleSave = async () => {
    if (!card) return

    const updates: CardUpdate = {}

    if (title !== card.title) updates.title = title
    if (description !== (card.description ?? "")) {
      updates.description = description === "" ? null : description
    }
    if (dueDateTouched) {
      updates.due_date = parseKanbanDateTimeLocalValue(dueDateInput)
    }
    if (priority !== (card.priority ?? null)) updates.priority = priority

    try {
      // Only save if there are changes
      if (Object.keys(updates).length > 0) {
        await onSave(card.id, updates)
      }
    } catch {
      return
    }

    setDueDateTouched(false)
    setIsDirty(false)
  }

  // Move-to-list: immediate move on select change
  const handleMoveToList = (targetListId: number) => {
    if (!card || targetListId === card.list_id) return
    onMove(card.id, targetListId)
  }

  const currentList = lists.find((l) => l.id === card?.list_id)

  const listOptions = lists.map((l) => ({
    value: l.id,
    label: l.name,
    disabled: l.id === card?.list_id
  }))

  return (
    <Drawer
      title={
        <div className="flex items-center justify-between">
          <span>Edit Card</span>
          {card && (
            <Space size={4}>
              <Button
                type="text"
                icon={<Copy className="w-4 h-4" />}
                onClick={() => copyCardMutation.mutate(card.id)}
                loading={copyCardMutation.isPending}
                title="Copy card"
              />
              {onArchive && (
                <Button
                  type="text"
                  icon={<Archive className="w-4 h-4" />}
                  onClick={() => onArchive(card.id, card.title)}
                  title="Archive card"
                />
              )}
              <Popconfirm
                title="Delete this card?"
                description="This card will be removed. This cannot be undone."
                onConfirm={() => onDelete(card.id)}
                okText="Delete"
                okType="danger"
              >
                <Button
                  danger
                  type="text"
                  icon={<Trash2 className="w-4 h-4" />}
                  title="Delete card"
                />
              </Popconfirm>
            </Space>
          )}
        </div>
      }
      open={open}
      onClose={onClose}
      styles={{ wrapper: { width: 400 } }}
      footer={
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button type="primary" onClick={handleSave} disabled={!isDirty}>
            Save Changes
          </Button>
        </div>
      }
    >
      {card && (
        <div className="space-y-4">
          {/* Title */}
          <div>
            <label className="block text-sm font-medium mb-1">Title</label>
            <Input
              value={title}
              onChange={(e) => {
                setTitle(e.target.value)
                setIsDirty(true)
              }}
              placeholder="Card title"
            />
          </div>

          {/* Current list indicator — immediate move on select */}
          <div>
            <label className="block text-sm font-medium mb-1">
              In list: <span className="font-normal">{currentList?.name}</span>
            </label>
            <Select
              placeholder="Move to..."
              style={{ width: "100%" }}
              value={card.list_id}
              onChange={handleMoveToList}
              options={listOptions}
            />
          </div>

          {/* Labels */}
          <div>
            <label className="block text-sm font-medium mb-1">Labels</label>
            {card.labels && card.labels.length > 0 && (
              <div className="flex gap-1 flex-wrap mb-2">
                {card.labels.map((label) => (
                  <span
                    key={label.id}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs text-white"
                    style={{ backgroundColor: label.color }}
                  >
                    {label.name}
                  </span>
                ))}
              </div>
            )}
            <LabelManager
              boardId={boardId}
              cardId={card.id}
              assignedLabelIds={(card.labels ?? []).map((l) => l.id)}
              onChanged={() => {
                // CardDetailPanel doesn't own the query, parent will invalidate
              }}
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Description
            </label>
            <Input.TextArea
              value={description}
              onChange={(e) => {
                setDescription(e.target.value)
                setIsDirty(true)
              }}
              placeholder="Add a description..."
              autoSize={{ minRows: 4, maxRows: 10 }}
            />
          </div>

          {/* Due Date */}
          <div>
            <label
              className="block text-sm font-medium mb-1"
              htmlFor="kanban-card-due-date"
            >
              Due Date
            </label>
            <input
              id="kanban-card-due-date"
              type="datetime-local"
              value={dueDateInput}
              onChange={(event) => {
                setDueDateInput(event.target.value)
                setDueDateTouched(true)
                setIsDirty(true)
              }}
              className="w-full rounded border border-border bg-background px-2 py-1 text-sm text-text focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>

          {/* Priority */}
          <div>
            <label className="block text-sm font-medium mb-1">Priority</label>
            <Select
              value={priority}
              onChange={(value) => {
                setPriority(value ?? null)
                setIsDirty(true)
              }}
              className="w-full"
              placeholder="Select priority"
              allowClear
              options={PRIORITY_OPTIONS.map((opt) => ({
                value: opt.value,
                label: (
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: opt.color }}
                    />
                    {opt.label}
                  </div>
                )
              }))}
            />
          </div>

          {/* Checklists */}
          <ChecklistSection cardId={card.id} />

          {/* Comments */}
          <div>
            <label className="block text-sm font-medium mb-2">Comments</label>
            {comments.length > 0 && (
              <div className="space-y-2 mb-3 max-h-60 overflow-y-auto">
                {comments.map((comment: Comment) => (
                  <div
                    key={comment.id}
                    className="bg-surface rounded p-2 text-sm"
                  >
                    <div className="text-text-muted text-xs mb-1">
                      {new Date(comment.created_at).toLocaleString(undefined, {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit"
                      })}
                    </div>
                    <div className="whitespace-pre-wrap">{comment.content}</div>
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <Input.TextArea
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                placeholder="Add a comment..."
                autoSize={{ minRows: 1, maxRows: 4 }}
                onPressEnter={(e) => {
                  if (e.shiftKey) return
                  e.preventDefault()
                  if (newComment.trim() && card) {
                    addCommentMutation.mutate({
                      cardId: card.id,
                      content: newComment.trim()
                    })
                  }
                }}
              />
              <Button
                type="primary"
                icon={<Send className="w-3.5 h-3.5" />}
                onClick={() => {
                  if (newComment.trim() && card) {
                    addCommentMutation.mutate({
                      cardId: card.id,
                      content: newComment.trim()
                    })
                  }
                }}
                loading={addCommentMutation.isPending}
                disabled={!newComment.trim()}
              />
            </div>
          </div>

          {/* Metadata (read-only info) */}
          <div className="pt-4 border-t text-xs text-text-muted space-y-1">
            <div>
              Created:{" "}
              {new Date(card.created_at).toLocaleString(undefined, {
                year: "numeric",
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit"
              })}
            </div>
            <div>
              Updated:{" "}
              {new Date(card.updated_at).toLocaleString(undefined, {
                year: "numeric",
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit"
              })}
            </div>
            <div>ID: {card.id}</div>
          </div>
        </div>
      )}
    </Drawer>
  )
}
