import React from "react"

import type { ChatLinkedResearchRun } from "@/services/tldw/TldwApiClient"
import type { ResearchFollowUpTarget } from "./research-chat-context"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"

import {
  CHAT_LINKED_RESEARCH_VISIBLE_TERMINAL_ROWS,
  getChatLinkedResearchActionPolicy,
  getChatLinkedResearchStatusLabel,
  isTerminalResearchRun,
  orderChatLinkedResearchRuns
} from "./research-run-status"

type ResearchRunStatusStackProps = {
  runs: ChatLinkedResearchRun[]
  onUseInChat?: (run: ChatLinkedResearchRun) => void
  onFollowUp?: (target: ResearchFollowUpTarget) => void
}

const STATUS_BADGE_VARIANT: Record<string, BadgeVariant> = {
  Running: "info",
  "Needs review": "warning",
  Completed: "success",
  Failed: "danger",
  Cancelled: "secondary",
  Paused: "warning"
}

export const ResearchRunStatusStack: React.FC<ResearchRunStatusStackProps> = ({
  runs,
  onUseInChat,
  onFollowUp
}) => {
  const [showAllTerminal, setShowAllTerminal] = React.useState(false)
  const orderedRuns = React.useMemo(() => orderChatLinkedResearchRuns(runs), [runs])
  const activeRuns = orderedRuns.filter((run) => !isTerminalResearchRun(run))
  const terminalRuns = orderedRuns.filter((run) => isTerminalResearchRun(run))
  const visibleTerminalRuns = showAllTerminal
    ? terminalRuns
    : terminalRuns.slice(0, CHAT_LINKED_RESEARCH_VISIBLE_TERMINAL_ROWS)
  const hiddenTerminalCount = Math.max(0, terminalRuns.length - visibleTerminalRuns.length)
  const visibleRuns = [...activeRuns, ...visibleTerminalRuns]

  React.useEffect(() => {
    setShowAllTerminal(false)
  }, [runs])

  if (runs.length === 0) {
    return null
  }

  return (
    <section
      aria-label="Linked deep research runs"
      data-testid="research-run-status-stack"
      className="mb-4 mt-4 w-full max-w-5xl px-4"
    >
      <div className="rounded-2xl border border-border/70 bg-surface/80 p-3 shadow-sm backdrop-blur-sm">
        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-text-subtle">
          Research runs
        </div>
        <div className="space-y-2">
          {visibleRuns.map((run) => {
            const statusLabel = getChatLinkedResearchStatusLabel(run)
            const actionPolicy = getChatLinkedResearchActionPolicy(run)
            return (
              <div
                key={run.run_id}
                data-testid="research-run-status-row"
                className="flex items-center justify-between gap-3 rounded-xl border border-border/60 bg-background/70 px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-text">{run.query}</div>
                  <div className="mt-1 flex items-center gap-2 text-xs text-text-subtle">
                    <Badge
                      data-testid={`research-run-status-badge-${run.run_id}`}
                      variant={STATUS_BADGE_VARIANT[statusLabel] ?? "secondary"}
                      size="sm"
                    >
                      {statusLabel}
                    </Badge>
                    {actionPolicy.reasonLabel ? <span>{actionPolicy.reasonLabel}</span> : null}
                    <span className="truncate">{run.run_id}</span>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  {actionPolicy.canUseInChat && (
                    <button
                      type="button"
                      className="text-sm font-medium text-text hover:text-primary"
                      onClick={() => onUseInChat?.(run)}
                    >
                      Use in Chat
                    </button>
                  )}
                  {actionPolicy.canFollowUp && (
                    <button
                      type="button"
                      className="text-sm font-medium text-text hover:text-primary"
                      onClick={() =>
                        onFollowUp?.({
                          run_id: run.run_id,
                          query: run.query
                        })
                      }
                    >
                      Follow up
                    </button>
                  )}
                  <a
                    href={actionPolicy.researchHref}
                    className="text-sm font-medium text-primary hover:underline"
                  >
                    {actionPolicy.primaryActionLabel}
                  </a>
                </div>
              </div>
            )
          })}
        </div>
        {hiddenTerminalCount > 0 && (
          <button
            type="button"
            className="mt-3 text-sm font-medium text-text-subtle hover:text-text"
            onClick={() => setShowAllTerminal((current) => !current)}
          >
            {showAllTerminal ? "Show fewer" : `Show ${hiddenTerminalCount} more`}
          </button>
        )}
      </div>
    </section>
  )
}
