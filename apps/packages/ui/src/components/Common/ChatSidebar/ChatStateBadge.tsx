import React from "react"
import type { TFunction } from "i18next"
import type { ConversationState } from "@/services/tldw/TldwApiClient"
import { normalizeConversationState } from "@/utils/conversation-state"
import { cn } from "@/libs/utils"

interface ChatStateBadgeProps {
  state?: ConversationState | string | null
  t?: TFunction
}

// Each state reads in words, with a dot in its own colour (CS-08).
const STATE_PRESENTATION: Record<
  ConversationState,
  { dotClassName: string; labelKey: string; defaultLabel: string }
> = {
  "in-progress": {
    dotClassName: "bg-primary",
    labelKey: "common:chatSidebar.state.inProgress",
    defaultLabel: "In progress"
  },
  resolved: {
    dotClassName: "bg-success",
    labelKey: "common:chatSidebar.state.resolved",
    defaultLabel: "Resolved"
  },
  backlog: {
    dotClassName: "bg-text-subtle",
    labelKey: "common:chatSidebar.state.backlog",
    defaultLabel: "Backlog"
  },
  "non-viable": {
    dotClassName: "bg-danger",
    labelKey: "common:chatSidebar.state.nonViable",
    defaultLabel: "Not viable"
  }
}

export function ChatStateBadge({ state, t }: ChatStateBadgeProps) {
  const displayState = normalizeConversationState(state)
  const presentation = STATE_PRESENTATION[displayState]
  const label = t
    ? String(t(presentation.labelKey, presentation.defaultLabel))
    : presentation.defaultLabel
  return (
    <span
      data-state={displayState}
      className="inline-flex shrink-0 items-center gap-1 text-[11px] font-medium text-text-muted">
      <span
        aria-hidden="true"
        className={cn("size-1.5 rounded-full", presentation.dotClassName)}
      />
      {label}
    </span>
  )
}
