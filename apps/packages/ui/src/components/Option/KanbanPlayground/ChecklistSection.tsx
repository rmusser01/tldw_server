import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  Alert,
  Button,
  Input,
  Checkbox,
  Progress,
  Collapse,
  Popconfirm,
  message
} from "antd"
import { Plus, Trash2 } from "lucide-react"

import type { ChecklistWithItems, ChecklistItem } from "@/types/kanban"
import {
  listChecklists,
  createChecklist,
  updateChecklist,
  deleteChecklist,
  createChecklistItem,
  updateChecklistItem,
  deleteChecklistItem,
  generateClientId
} from "@/services/kanban"

interface ChecklistSectionProps {
  cardId: number
}

export const ChecklistSection = ({ cardId }: ChecklistSectionProps) => {
  const queryClient = useQueryClient()
  const queryKey = ["kanban-card-checklists", cardId]

  const { data: checklists = [], isError } = useQuery({
    queryKey,
    queryFn: () => listChecklists(cardId),
    staleTime: 20 * 1000
  })

  const [addingChecklist, setAddingChecklist] = useState(false)
  const [newChecklistTitle, setNewChecklistTitle] = useState("")

  const invalidate = () => queryClient.invalidateQueries({ queryKey })

  const createChecklistMutation = useMutation({
    mutationFn: (title: string) =>
      createChecklist(cardId, { title, client_id: generateClientId() }),
    onSuccess: () => {
      invalidate()
      setAddingChecklist(false)
      setNewChecklistTitle("")
    },
    onError: () => {
      message.error("Failed to create checklist. Please try again.")
    }
  })

  const handleAddChecklist = () => {
    if (isError) return
    const trimmed = newChecklistTitle.trim()
    if (!trimmed) return
    createChecklistMutation.mutate(trimmed)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium">Checklists</label>
      </div>

      {isError && (
        <Alert type="error" showIcon title="Unable to load checklists." />
      )}

      {checklists.map((cl) => (
        <SingleChecklist key={cl.id} checklist={cl} onChanged={invalidate} />
      ))}

      {addingChecklist ? (
        <div className="space-y-2">
          <Input
            size="small"
            placeholder="Checklist title"
            value={newChecklistTitle}
            onChange={(e) => setNewChecklistTitle(e.target.value)}
            onPressEnter={handleAddChecklist}
            autoFocus
          />
          <div className="flex gap-2">
            <Button
              size="small"
              type="primary"
              onClick={handleAddChecklist}
              loading={createChecklistMutation.isPending}
              disabled={isError}
            >
              Add
            </Button>
            <Button size="small" onClick={() => setAddingChecklist(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button
          size="small"
          type="dashed"
          icon={<Plus className="w-3.5 h-3.5" />}
          onClick={() => setAddingChecklist(true)}
          disabled={isError}
        >
          Add Checklist
        </Button>
      )}
    </div>
  )
}

// Single checklist with items
const SingleChecklist = ({
  checklist,
  onChanged
}: {
  checklist: ChecklistWithItems
  onChanged: () => void
}) => {
  const [editingTitle, setEditingTitle] = useState(false)
  const [titleValue, setTitleValue] = useState(checklist.title)
  const [addingItem, setAddingItem] = useState(false)
  const [newItemContent, setNewItemContent] = useState("")

  const complete = checklist.items.filter((i) => i.checked).length
  const total = checklist.items.length
  const percent = total > 0 ? Math.round((complete / total) * 100) : 0

  const updateTitleMutation = useMutation({
    mutationFn: (title: string) => updateChecklist(checklist.id, { title }),
    onSuccess: () => {
      setEditingTitle(false)
      onChanged()
    },
    onError: () => {
      message.error("Failed to rename checklist. Please try again.")
    }
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteChecklist(checklist.id),
    onSuccess: onChanged,
    onError: () => {
      message.error("Failed to delete checklist. Please try again.")
    }
  })

  const addItemMutation = useMutation({
    mutationFn: (content: string) =>
      createChecklistItem(checklist.id, {
        content,
        client_id: generateClientId()
      }),
    onSuccess: () => {
      setNewItemContent("")
      onChanged()
    },
    onError: () => {
      message.error("Failed to add item. Please try again.")
    }
  })

  const toggleItemMutation = useMutation({
    mutationFn: ({ itemId, checked }: { itemId: number; checked: boolean }) =>
      updateChecklistItem(itemId, { checked }),
    onSuccess: onChanged,
    onError: () => {
      message.error("Failed to update item. Please try again.")
    }
  })

  const updateItemContentMutation = useMutation({
    mutationFn: ({ itemId, content }: { itemId: number; content: string }) =>
      updateChecklistItem(itemId, { content }),
    onSuccess: onChanged,
    onError: () => {
      message.error("Failed to update item. Please try again.")
    }
  })

  const deleteItemMutation = useMutation({
    mutationFn: (itemId: number) => deleteChecklistItem(itemId),
    onSuccess: onChanged,
    onError: () => {
      message.error("Failed to delete item. Please try again.")
    }
  })

  const handleAddItem = () => {
    const trimmed = newItemContent.trim()
    if (!trimmed) return
    addItemMutation.mutate(trimmed)
  }

  return (
    <div className="border rounded-md p-2 bg-surface">
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        {editingTitle ? (
          <Input
            size="small"
            value={titleValue}
            onChange={(e) => setTitleValue(e.target.value)}
            onPressEnter={() => {
              const t = titleValue.trim()
              if (t && t !== checklist.title) updateTitleMutation.mutate(t)
              else setEditingTitle(false)
            }}
            onBlur={() => {
              const t = titleValue.trim()
              if (t && t !== checklist.title) updateTitleMutation.mutate(t)
              else setEditingTitle(false)
            }}
            autoFocus
            className="flex-1 mr-2"
          />
        ) : (
          <span
            className="text-sm font-medium cursor-pointer"
            onDoubleClick={() => {
              setTitleValue(checklist.title)
              setEditingTitle(true)
            }}
          >
            {checklist.title}
          </span>
        )}
        <Popconfirm
          title="Delete this checklist?"
          onConfirm={() => deleteMutation.mutate()}
          okText="Delete"
          okType="danger"
        >
          <Button
            type="text"
            size="small"
            danger
            icon={<Trash2 className="w-3 h-3" />}
          />
        </Popconfirm>
      </div>

      {/* Progress */}
      {total > 0 && (
        <div className="mb-2">
          <Progress
            percent={percent}
            size="small"
            format={() => `${complete}/${total}`}
            status={complete === total ? "success" : "active"}
          />
        </div>
      )}

      {/* Items */}
      <div className="space-y-1">
        {checklist.items.map((item) => (
          <ChecklistItemRow
            key={item.id}
            item={item}
            onToggle={(checked) =>
              toggleItemMutation.mutate({ itemId: item.id, checked })
            }
            onUpdateContent={(content) =>
              updateItemContentMutation.mutate({ itemId: item.id, content })
            }
            onDelete={() => deleteItemMutation.mutate(item.id)}
          />
        ))}
      </div>

      {/* Add item */}
      {addingItem ? (
        <div className="mt-1 space-y-1">
          <Input
            size="small"
            placeholder="Add an item"
            value={newItemContent}
            onChange={(e) => setNewItemContent(e.target.value)}
            onPressEnter={handleAddItem}
            onKeyDown={(e) => {
              if (e.key === "Escape") setAddingItem(false)
            }}
            autoFocus
          />
          <div className="flex gap-1">
            <Button
              size="small"
              type="primary"
              onClick={handleAddItem}
              loading={addItemMutation.isPending}
            >
              Add
            </Button>
            <Button size="small" onClick={() => setAddingItem(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button
          type="text"
          size="small"
          icon={<Plus className="w-3 h-3" />}
          onClick={() => setAddingItem(true)}
          className="mt-1 text-text-muted"
        >
          Add item
        </Button>
      )}
    </div>
  )
}

// Single checklist item row
const ChecklistItemRow = ({
  item,
  onToggle,
  onUpdateContent,
  onDelete
}: {
  item: ChecklistItem
  onToggle: (checked: boolean) => void
  onUpdateContent: (content: string) => void
  onDelete: () => void
}) => {
  const [editing, setEditing] = useState(false)
  const [editValue, setEditValue] = useState(item.content)

  return (
    <div className="flex items-center gap-1 group">
      <Checkbox
        checked={item.checked}
        onChange={(e) => onToggle(e.target.checked)}
      />
      {editing ? (
        <Input
          size="small"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onPressEnter={() => {
            const t = editValue.trim()
            if (t && t !== item.content) onUpdateContent(t)
            setEditing(false)
          }}
          onBlur={() => {
            const t = editValue.trim()
            if (t && t !== item.content) onUpdateContent(t)
            setEditing(false)
          }}
          autoFocus
          className="flex-1"
        />
      ) : (
        <span
          className={`flex-1 text-xs cursor-pointer ${
            item.checked ? "line-through text-text-muted" : ""
          }`}
          onDoubleClick={() => {
            setEditValue(item.content)
            setEditing(true)
          }}
        >
          {item.content}
        </span>
      )}
      <Button
        type="text"
        size="small"
        danger
        className="!p-0 !w-5 !h-5 opacity-0 group-hover:opacity-100 transition-opacity"
        icon={<Trash2 className="w-3 h-3" />}
        onClick={onDelete}
      />
    </div>
  )
}
