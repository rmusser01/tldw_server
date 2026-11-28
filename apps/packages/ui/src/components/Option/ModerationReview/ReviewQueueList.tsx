import React from "react"

import type { ModerationReviewItem } from "@/services/moderation"
import {
  formatReviewDate,
  getReviewItemSourceLabel,
  getReviewItemUserLabel,
  isExecutionIncident,
  reviewPhaseLabel,
  REVIEW_STATUS_LABELS,
  SEVERITY_LABELS
} from "./review-utils"

type ReviewQueueListProps = {
  items: ModerationReviewItem[]
  selectedItemId: string | null
  selectedForBulkIds?: string[]
  onSelect: (itemId: string) => void
  onToggleSelected?: (itemId: string) => void
}

export const ReviewQueueList: React.FC<ReviewQueueListProps> = ({
  items,
  selectedItemId,
  selectedForBulkIds = [],
  onSelect,
  onToggleSelected
}) => {
  const bulkSelectionEnabled = Boolean(onToggleSelected)
  const headerColumns = bulkSelectionEnabled
    ? "[@container(min-width:800px)]:grid-cols-[36px_90px_65px_110px_104px_minmax(120px,1fr)_130px]"
    : "[@container(min-width:800px)]:grid-cols-[90px_65px_110px_104px_minmax(120px,1fr)_130px]"
  const rowColumns = bulkSelectionEnabled ? "[@container(min-width:800px)]:grid-cols-[36px_minmax(0,1fr)]" : "grid-cols-1"

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-surface [container-type:inline-size]">
      <div className={`hidden gap-3 border-b border-border bg-surface2 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-text-muted [@container(min-width:800px)]:grid ${headerColumns}`}>
        {bulkSelectionEnabled && <div><span className="sr-only">Select</span></div>}
        <div>Status</div>
        <div>Severity</div>
        <div>Category</div>
        <div>Phase</div>
        <div>Excerpt</div>
        <div>Created</div>
      </div>
      <div role="list" aria-label="Moderation review items">
        {items.map((item) => {
          const selected = item.id === selectedItemId
          const selectedForBulk = selectedForBulkIds.includes(item.id)
          return (
            <div
              key={item.id}
              role="listitem"
              className={`grid gap-3 border-b border-border px-3 py-3 last:border-b-0 [@container(min-width:800px)]:items-center ${rowColumns} ${
                selected ? "bg-blue-50 text-blue-950 dark:bg-blue-950/30 dark:text-blue-100" : "bg-surface hover:bg-surface2"
              }`}
            >
              {bulkSelectionEnabled && (
                <div className="flex items-start">
                  <input
                    type="checkbox"
                    checked={selectedForBulk}
                    onChange={() => onToggleSelected?.(item.id)}
                    aria-label={`Select review item ${item.id}`}
                    className="mt-1 h-4 w-4 rounded border-border text-blue-600 focus:ring-2 focus:ring-blue-500/30"
                  />
                </div>
              )}
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                className="grid min-w-0 w-full gap-3 break-words text-left text-sm [@container(min-width:800px)]:grid-cols-[90px_65px_110px_104px_minmax(120px,1fr)_130px] [@container(min-width:800px)]:items-center"
                aria-current={selected ? "true" : undefined}
              >
                <div>
                  <div className="font-medium text-text">{REVIEW_STATUS_LABELS[item.status]}</div>
                  <div className="mt-0.5 text-xs text-text-muted [@container(min-width:800px)]:hidden">{getReviewItemSourceLabel(item)}</div>
                </div>
                <div className="text-text-muted">{item.severity ? SEVERITY_LABELS[item.severity] : "Unknown"}</div>
                <div className="min-w-0 text-text-muted">{isExecutionIncident(item) ? "Operational failure" : item.category || "Uncategorized"}</div>
                <div className="text-text-muted">{reviewPhaseLabel(item.phase)}</div>
                <div className="min-w-0">
                  <div className="truncate text-text">{item.excerpt}</div>
                  <div className="mt-0.5 truncate text-xs text-text-muted">
                    {getReviewItemSourceLabel(item)}{!isExecutionIncident(item) ? ` · ${getReviewItemUserLabel(item)}` : ""}
                    {item.recommended_action ? ` · Recommended: ${item.recommended_action}` : ""}
                  </div>
                </div>
                <div className="text-xs text-text-muted">{formatReviewDate(item.created_at)}</div>
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
