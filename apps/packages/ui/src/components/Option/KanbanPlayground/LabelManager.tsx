import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Popover, Button, Input, Checkbox, Popconfirm, message } from "antd"
import { Plus, Edit2, Trash2, Tag } from "lucide-react"

import { generateClientId } from "@/services/kanban"
import { useKanbanScope } from "./KanbanScope"
import type { Label } from "@/types/kanban"

const PRESET_COLORS: { hex: string; name: string }[] = [
  { hex: "#ef4444", name: "Red" },
  { hex: "#f97316", name: "Orange" },
  { hex: "#eab308", name: "Yellow" },
  { hex: "#22c55e", name: "Green" },
  { hex: "#3b82f6", name: "Blue" },
  { hex: "#8b5cf6", name: "Violet" },
  { hex: "#ec4899", name: "Pink" },
  { hex: "#6b7280", name: "Gray" }
]

interface LabelManagerProps {
  boardId: number
  cardId: number
  assignedLabelIds: number[]
  onChanged?: () => void
  children?: React.ReactNode
}

export const LabelManager = ({
  boardId,
  cardId,
  assignedLabelIds,
  onChanged,
  children
}: LabelManagerProps) => {
  const queryClient = useQueryClient()
  const { api, key, isActive } = useKanbanScope()
  const { listLabels, createLabel, updateLabel, deleteLabel, assignLabelToCard, removeLabelFromCard } = api
  const [open, setOpen] = useState(false)

  // Creating new label
  const [newLabelName, setNewLabelName] = useState("")
  const [newLabelColor, setNewLabelColor] = useState(PRESET_COLORS[0].hex)

  // Editing existing label
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editName, setEditName] = useState("")
  const [editColor, setEditColor] = useState("")

  const { data: labels = [] } = useQuery({
    queryKey: key("kanban-board-labels", boardId),
    queryFn: () => listLabels(boardId),
    enabled: open,
    staleTime: 30 * 1000
  })

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: key("kanban-board-labels", boardId) })
    queryClient.invalidateQueries({ queryKey: key("kanban-board", boardId) })
    onChanged?.()
  }

  const createMutation = useMutation({
    mutationFn: () =>
      createLabel(boardId, {
        name: newLabelName.trim(),
        color: newLabelColor,
        client_id: generateClientId()
      }),
    onSuccess: () => {
      if (!isActive()) return
      setNewLabelName("")
      invalidate()
    },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to create label. Please try again.")
    }
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: { name?: string; color?: string } }) =>
      updateLabel(id, data),
    onSuccess: () => {
      if (!isActive()) return
      setEditingId(null)
      invalidate()
    },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to update label. Please try again.")
    }
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteLabel(id),
    onSuccess: () => { if (isActive()) invalidate() },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to delete label. Please try again.")
    }
  })

  const assignMutation = useMutation({
    mutationFn: (labelId: number) => assignLabelToCard(cardId, labelId),
    onSuccess: () => { if (isActive()) invalidate() },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to update label assignment. Please try again.")
    }
  })

  const removeMutation = useMutation({
    mutationFn: (labelId: number) => removeLabelFromCard(cardId, labelId),
    onSuccess: () => { if (isActive()) invalidate() },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to update label assignment. Please try again.")
    }
  })

  const handleToggle = (labelId: number, isAssigned: boolean) => {
    if (isAssigned) {
      removeMutation.mutate(labelId)
    } else {
      assignMutation.mutate(labelId)
    }
  }

  const handleCreate = () => {
    if (!newLabelName.trim()) return
    createMutation.mutate()
  }

  const startEdit = (label: Label) => {
    setEditingId(label.id)
    setEditName(label.name)
    setEditColor(label.color)
  }

  const confirmEdit = () => {
    if (!editingId) return
    updateMutation.mutate({
      id: editingId,
      data: { name: editName.trim() || undefined, color: editColor || undefined }
    })
  }

  const content = (
    <div className="w-64 space-y-3">
      <div className="text-sm font-medium">Labels</div>

      {/* Existing labels */}
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {labels.map((label) => {
          const isAssigned = assignedLabelIds.includes(label.id)

          if (editingId === label.id) {
            return (
              <div key={label.id} className="space-y-1">
                <Input
                  size="small"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onPressEnter={confirmEdit}
                  autoFocus
                />
                <div className="flex gap-1" role="radiogroup" aria-label="Label color">
                  {PRESET_COLORS.map((c) => (
                    <button
                      key={c.hex}
                      role="radio"
                      aria-checked={editColor === c.hex}
                      className={`w-5 h-5 rounded-sm border-2 ${
                        editColor === c.hex
                          ? "border-text"
                          : "border-transparent"
                      }`}
                      style={{ backgroundColor: c.hex }}
                      onClick={() => setEditColor(c.hex)}
                      aria-label={`${c.name}${editColor === c.hex ? " (selected)" : ""}`}
                      title={c.name}
                    />
                  ))}
                </div>
                <div className="flex gap-1">
                  <Button size="small" type="primary" onClick={confirmEdit}>
                    Save
                  </Button>
                  <Button size="small" onClick={() => setEditingId(null)}>
                    Cancel
                  </Button>
                </div>
              </div>
            )
          }

          return (
            <div
              key={label.id}
              className="flex items-center gap-2 p-1 rounded hover:bg-surface"
            >
              <Checkbox
                checked={isAssigned}
                onChange={() => handleToggle(label.id, isAssigned)}
              />
              <span
                className="flex-1 text-xs px-2 py-0.5 rounded text-white truncate"
                style={{ backgroundColor: label.color }}
              >
                {label.name}
              </span>
              <Button
                type="text"
                size="small"
                className="!p-0 !w-5 !h-5"
                icon={<Edit2 className="w-3 h-3" />}
                onClick={() => startEdit(label)}
              />
              <Popconfirm
                title={`Delete label "${label.name}"?`}
                onConfirm={() => deleteMutation.mutate(label.id)}
                okText="Delete"
                okType="danger"
              >
                <Button
                  type="text"
                  size="small"
                  danger
                  className="!p-0 !w-5 !h-5"
                  icon={<Trash2 className="w-3 h-3" />}
                />
              </Popconfirm>
            </div>
          )
        })}
      </div>

      {/* Create new label */}
      <div className="border-t pt-2 space-y-1">
        <Input
          size="small"
          placeholder="New label name"
          value={newLabelName}
          onChange={(e) => setNewLabelName(e.target.value)}
          onPressEnter={handleCreate}
        />
        <div className="flex gap-1" role="radiogroup" aria-label="Label color">
          {PRESET_COLORS.map((c) => (
            <button
              key={c.hex}
              role="radio"
              aria-checked={newLabelColor === c.hex}
              className={`w-5 h-5 rounded-sm border-2 ${
                newLabelColor === c.hex
                  ? "border-text"
                  : "border-transparent"
              }`}
              style={{ backgroundColor: c.hex }}
              onClick={() => setNewLabelColor(c.hex)}
              aria-label={`${c.name}${newLabelColor === c.hex ? " (selected)" : ""}`}
              title={c.name}
            />
          ))}
        </div>
        <Button
          size="small"
          type="primary"
          icon={<Plus className="w-3 h-3" />}
          onClick={handleCreate}
          disabled={!newLabelName.trim()}
          loading={createMutation.isPending}
        >
          Create Label
        </Button>
      </div>
    </div>
  )

  return (
    <Popover
      content={content}
      trigger="click"
      open={open}
      onOpenChange={setOpen}
      placement="bottomLeft"
    >
      {children ?? (
        <Button
          size="small"
          type="dashed"
          icon={<Tag className="w-3.5 h-3.5" />}
        >
          Add Label
        </Button>
      )}
    </Popover>
  )
}
