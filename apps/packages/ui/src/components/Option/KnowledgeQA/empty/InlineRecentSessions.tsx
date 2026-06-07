import React from "react"
import { Clock3, FileText, Sparkles } from "lucide-react"
import { cn } from "@/libs/utils"
import type { SearchHistoryItem } from "../types"
import { getKnowledgeAnswerTrustLabel } from "../trustState"

type InlineRecentSessionsProps = {
  items: SearchHistoryItem[]
  onRestore: (item: SearchHistoryItem) => void
  className?: string
}

const MAX_VISIBLE_SESSIONS = 5

function formatRelativeTime(timestamp: string): string {
  const parsed = new Date(timestamp).getTime()
  if (Number.isNaN(parsed)) return "Unknown"
  const diff = Date.now() - parsed
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return "Just now"
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days === 1) return "Yesterday"
  if (days < 7) return `${days}d ago`
  return new Date(timestamp).toLocaleDateString([], { month: "short", day: "numeric" })
}

export function InlineRecentSessions({
  items,
  onRestore,
  className,
}: InlineRecentSessionsProps) {
  if (items.length === 0) return null

  const visible = items.slice(0, MAX_VISIBLE_SESSIONS)

  return (
    <div className={cn("mx-auto max-w-2xl", className)}>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
        Recent searches
      </p>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
        {visible.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => onRestore(item)}
            className="flex min-w-[180px] max-w-[240px] flex-shrink-0 flex-col gap-1 rounded-lg border border-border/80 bg-surface px-3 py-2 text-left transition-colors hover:border-primary/30 hover:bg-surface2"
          >
            <p className="truncate text-sm font-medium text-text">
              {item.query}
            </p>
            <div className="flex items-center gap-2 text-[11px] text-text-muted">
              <span className="inline-flex items-center gap-0.5">
                <FileText className="h-3 w-3" />
                {item.sourcesCount}
              </span>
              {item.hasAnswer && (
                <span className="inline-flex items-center gap-0.5">
                  <Sparkles className="h-3 w-3" />
                </span>
              )}
              {item.trustState ? (
                <span className="inline-flex min-w-0 truncate">
                  {getKnowledgeAnswerTrustLabel(item.trustState)}
                </span>
              ) : null}
              <span className="inline-flex items-center gap-0.5">
                <Clock3 className="h-3 w-3" />
                {formatRelativeTime(item.timestamp)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
