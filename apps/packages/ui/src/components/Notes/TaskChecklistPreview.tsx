import React from "react"
import type { NoteTask, NoteTaskStatus } from "@/services/notes-tasks"
import {
  getNextTaskStatus,
  parseChecklistItems,
  stripChecklistMetadataForLabel,
  type ParsedChecklistItem
} from "@/components/Notes/task-markdown"

export type TaskChecklistTogglePayload = {
  item: ParsedChecklistItem
  task?: NoteTask
  lineNumber: number
  checked: boolean
  nextStatus: NoteTaskStatus
}

export interface TaskChecklistPreviewProps {
  content: string
  tasks?: NoteTask[]
  isDirty?: boolean
  disabled?: boolean
  compact?: boolean
  onToggleLocal?: (payload: TaskChecklistTogglePayload) => void
  onToggleTaskStatus?: (payload: TaskChecklistTogglePayload) => void
}

const PROJECTION_LABELS: Partial<Record<NoteTask["projection_status"], string>> = {
  ambiguous: "Ambiguous",
  unlinked: "Unlinked",
  deleted: "Deleted"
}

const taskLineNumber = (task: NoteTask): number | null => {
  const lineNumber = Number(task.projection?.line_number)
  return Number.isFinite(lineNumber) && lineNumber > 0 ? Math.trunc(lineNumber) : null
}

const buildTaskLineMap = (tasks: NoteTask[]): Map<number, NoteTask> => {
  const out = new Map<number, NoteTask>()
  for (const task of tasks) {
    const lineNumber = taskLineNumber(task)
    if (lineNumber == null || out.has(lineNumber)) continue
    out.set(lineNumber, task)
  }
  return out
}

const TaskChecklistPreview: React.FC<TaskChecklistPreviewProps> = ({
  content,
  tasks = [],
  isDirty = false,
  disabled = false,
  compact = false,
  onToggleLocal,
  onToggleTaskStatus
}) => {
  const items = React.useMemo(() => parseChecklistItems(content), [content])
  const taskByLine = React.useMemo(() => buildTaskLineMap(tasks), [tasks])

  if (items.length === 0) return null

  return (
    <div className={`notes-task-checklist-preview${compact ? " is-compact" : ""}`}>
      <ul className="notes-task-checklist-preview__list">
        {items.map((item) => {
          const task = taskByLine.get(item.lineNumber)
          const checked = task ? task.status === "done" : item.checked
          const label = stripChecklistMetadataForLabel(task?.text || item.text)
          const nextStatus = getNextTaskStatus(checked)
          const projectionLabel = task?.projection_status
            ? PROJECTION_LABELS[task.projection_status]
            : null
          const toggleDisabled = disabled || (!isDirty && !task)
          const payload: TaskChecklistTogglePayload = {
            item,
            task,
            lineNumber: item.lineNumber,
            checked,
            nextStatus
          }

          return (
            <li key={`${item.lineNumber}:${item.rawLine}`} className="notes-task-checklist-preview__item">
              <label className="notes-task-checklist-preview__row">
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={toggleDisabled}
                  aria-label={label ? `Task: ${label}` : `Task on line ${item.lineNumber}`}
                  onChange={() => {
                    if (isDirty || !task) {
                      onToggleLocal?.(payload)
                      return
                    }
                    onToggleTaskStatus?.(payload)
                  }}
                />
                <span className="notes-task-checklist-preview__text">{label || item.text}</span>
              </label>
              <span className="notes-task-checklist-preview__badges" aria-live="polite">
                {projectionLabel ? (
                  <span className="notes-task-checklist-preview__badge">{projectionLabel}</span>
                ) : null}
                {item.hasChildContent ? (
                  <span className="notes-task-checklist-preview__badge">Has details</span>
                ) : null}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default TaskChecklistPreview
