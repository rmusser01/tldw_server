import React from "react"
import { Tooltip } from "antd"
import { ListChecks } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { NoteTask } from "@/services/notes-tasks"
import { parseChecklistItems } from "@/components/Notes/task-markdown"
import TaskChecklistPreview, {
  buildTaskLineMap,
  type TaskChecklistPreviewProps
} from "@/components/Notes/TaskChecklistPreview"

/** Done and total checklist items, preferring server task status per line. */
export const summarizeChecklist = (
  content: string,
  tasks: NoteTask[] = []
): { done: number; total: number } => {
  const items = parseChecklistItems(content)
  const taskByLine = buildTaskLineMap(tasks)
  let done = 0
  for (const item of items) {
    const task = taskByLine.get(item.lineNumber)
    if (task ? task.status === "done" : item.checked) done += 1
  }
  return { done, total: items.length }
}

/**
 * A compact "Tasks 1/2" chip above the preview (NE-09). The rendered
 * preview already shows the checklist, so the checkable list stays folded
 * into the chip until asked for instead of repeating every item.
 */
const TaskChecklistToggle: React.FC<TaskChecklistPreviewProps> = (props) => {
  const { t } = useTranslation(["option"])
  const [open, setOpen] = React.useState(false)
  const listId = React.useId()
  const { done, total } = React.useMemo(
    () => summarizeChecklist(props.content, props.tasks),
    [props.content, props.tasks]
  )

  if (total === 0) return null

  return (
    <div className="flex flex-col gap-2" data-testid="notes-task-checklist-toggle">
      <Tooltip
        title={t("option:notesSearch.taskChipTooltip", {
          defaultValue: "Checking a box updates the note text and your task list."
        })}
      >
        <button
          type="button"
          className="inline-flex w-fit items-center gap-1 rounded-full border border-border bg-surface px-2 py-0.5 text-xs text-text-muted transition-colors hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          aria-expanded={open}
          aria-controls={listId}
          onClick={() => setOpen((current) => !current)}
          data-testid="notes-task-chip"
        >
          <ListChecks className="size-3.5" aria-hidden="true" />
          {`${t("option:notesSearch.taskChipLabel", { defaultValue: "Tasks" })} ${done}/${total}`}
        </button>
      </Tooltip>
      <div id={listId} hidden={!open}>
        {open ? <TaskChecklistPreview {...props} /> : null}
      </div>
    </div>
  )
}

export default TaskChecklistToggle
