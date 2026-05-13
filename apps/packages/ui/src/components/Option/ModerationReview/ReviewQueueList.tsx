import React from "react"

import type { ModerationReviewItem } from "@/services/moderation"
import {
  formatReviewDate,
  getReviewItemSourceLabel,
  getReviewItemUserLabel,
  REVIEW_STATUS_LABELS,
  SEVERITY_LABELS
} from "./review-utils"

type ReviewQueueListProps = {
  items: ModerationReviewItem[]
  selectedItemId: string | null
  onSelect: (itemId: string) => void
}

export const ReviewQueueList: React.FC<ReviewQueueListProps> = ({ items, selectedItemId, onSelect }) => {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="hidden grid-cols-[110px_90px_120px_80px_1fr_150px] gap-3 border-b border-border bg-surface2 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-text-muted lg:grid">
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
          return (
            <div key={item.id} role="listitem" className="border-b border-border last:border-b-0">
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                className={`grid w-full gap-2 px-3 py-3 text-left text-sm lg:grid-cols-[110px_90px_120px_80px_1fr_150px] lg:items-center ${
                  selected ? "bg-blue-50 text-blue-950 dark:bg-blue-950/30 dark:text-blue-100" : "bg-surface hover:bg-surface2"
                }`}
                aria-current={selected ? "true" : undefined}
              >
                <div>
                  <div className="font-medium text-text">{REVIEW_STATUS_LABELS[item.status]}</div>
                  <div className="mt-0.5 text-xs text-text-muted lg:hidden">{getReviewItemSourceLabel(item)}</div>
                </div>
                <div className="text-text-muted">{item.severity ? SEVERITY_LABELS[item.severity] : "Unknown"}</div>
                <div className="text-text-muted">{item.category || "Uncategorized"}</div>
                <div className="capitalize text-text-muted">{item.phase}</div>
                <div className="min-w-0">
                  <div className="truncate text-text">{item.excerpt}</div>
                  <div className="mt-0.5 truncate text-xs text-text-muted">
                    {getReviewItemSourceLabel(item)} · {getReviewItemUserLabel(item)}
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
