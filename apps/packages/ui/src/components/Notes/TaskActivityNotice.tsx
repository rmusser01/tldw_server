import React from "react"
import { Button } from "antd"
import { useTranslation } from "react-i18next"
import type { NoteTaskActivityEvent } from "@/services/notes-tasks"

export type TaskActivityNoticeProps = {
  events: NoteTaskActivityEvent[]
  noteTitle?: string | null
  testId?: string
  compact?: boolean
  dismissingEventId?: string | null
  onInspect?: () => void
  onDismiss?: (eventId: string) => void
}

const eventActorLabel = (event: NoteTaskActivityEvent): string => {
  const actorId = String(event.actor_id || "").trim()
  if (actorId) return actorId
  const actorType = String(event.actor_type || "").trim()
  return actorType || "Agent"
}

const eventToolLabel = (event: NoteTaskActivityEvent): string => {
  const toolName = String(event.tool_name || "").trim()
  if (toolName) return toolName
  return String(event.event_type || "task activity").replace(/_/g, " ")
}

const taskCountLabel = (count: number): string => `${count} task${count === 1 ? "" : "s"}`

const TaskActivityNotice: React.FC<TaskActivityNoticeProps> = ({
  events,
  noteTitle,
  testId = "notes-task-activity-notice",
  compact = false,
  dismissingEventId = null,
  onInspect,
  onDismiss
}) => {
  const { t } = useTranslation(["option", "common"])
  const firstEvent = events[0]
  if (!firstEvent) return null

  const actorLabel = eventActorLabel(firstEvent)
  const toolLabel = eventToolLabel(firstEvent)
  const affectedLabel = noteTitle?.trim() || t("option:taskActivity.thisNote", {
    defaultValue: "this note"
  })
  const countLabel = taskCountLabel(events.length)
  const summaryDefault = `${actorLabel} via ${toolLabel} changed ${countLabel} in ${affectedLabel}.`
  const summary = t("option:taskActivity.summary", {
    defaultValue: summaryDefault,
    actor: actorLabel,
    tool: toolLabel,
    count: countLabel,
    note: affectedLabel
  })

  return (
    <div
      className={
        compact
          ? "mt-2 rounded-md border border-warning/40 bg-warning/10 px-2 py-1 text-[11px] text-warning"
          : "mt-2 rounded border border-primary/40 bg-primary/10 px-2 py-2 text-[12px] text-primary"
      }
      role="status"
      aria-live="polite"
      data-testid={testId}
    >
      <div className="font-medium">{summary}</div>
      <div className="mt-1 flex items-center gap-2">
        {onInspect ? (
          <Button
            size="small"
            type="link"
            className="h-auto !px-0 text-[12px]"
            aria-label={t("option:taskActivity.inspectAria", {
              defaultValue: "Inspect task activity"
            })}
            onClick={onInspect}
          >
            {t("option:taskActivity.inspect", { defaultValue: "Inspect" })}
          </Button>
        ) : null}
        {onDismiss ? (
          <Button
            size="small"
            type="link"
            className="h-auto !px-0 text-[12px]"
            loading={dismissingEventId === firstEvent.id}
            aria-label={t("option:taskActivity.dismissAria", {
              defaultValue: "Dismiss task activity"
            })}
            onClick={() => onDismiss(firstEvent.id)}
          >
            {t("option:taskActivity.dismiss", { defaultValue: "Dismiss" })}
          </Button>
        ) : null}
      </div>
    </div>
  )
}

export default TaskActivityNotice
