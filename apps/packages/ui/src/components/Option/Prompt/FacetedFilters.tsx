import React, { useState } from "react"
import type { TagMatchMode } from "./custom-prompts-utils"

type Props = {
  typeFilter: string
  onTypeFilterChange: (v: string) => void
  typeCounts: Record<string, number>
  syncFilter: string
  onSyncFilterChange: (v: string) => void
  syncCounts: Record<string, number>
  tagFilter: string[]
  onTagFilterChange: (v: string[]) => void
  tagMatchMode: TagMatchMode
  onTagMatchModeChange: (v: TagMatchMode) => void
  tagCounts: Record<string, number>
}

const TYPE_OPTIONS = [
  { value: "all", label: "All" },
  { value: "system", label: "System" },
  { value: "quick", label: "Quick" },
  { value: "mixed", label: "Mixed" },
]

const SYNC_OPTIONS = [
  { value: "all", label: "All" },
  { value: "local", label: "Local" },
  { value: "synced", label: "Synced" },
  { value: "pending", label: "Pending" },
  { value: "conflict", label: "Conflict" },
]

const INITIAL_TAG_LIMIT = 5

export const FacetedFilters: React.FC<Props> = ({
  typeFilter,
  onTypeFilterChange,
  typeCounts,
  syncFilter,
  onSyncFilterChange,
  syncCounts,
  tagFilter,
  onTagFilterChange,
  tagMatchMode,
  onTagMatchModeChange,
  tagCounts,
}) => {
  const [showAllTags, setShowAllTags] = useState(false)
  const sortedTags = Object.entries(tagCounts).sort((a, b) => b[1] - a[1])
  const visibleTags = showAllTags
    ? sortedTags
    : sortedTags.slice(0, INITIAL_TAG_LIMIT)
  const hiddenCount = sortedTags.length - INITIAL_TAG_LIMIT

  const toggleTag = (tag: string) => {
    if (tagFilter.includes(tag)) {
      onTagFilterChange(tagFilter.filter((t) => t !== tag))
    } else {
      onTagFilterChange([...tagFilter, tag])
    }
  }

  return (
    <div className="space-y-4" data-testid="faceted-filters">
      {/* Type filter */}
      <div>
        <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">
          Type
        </h4>
        <div className="space-y-0.5">
          {TYPE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              data-testid={`facet-type-${opt.value}`}
              onClick={() => onTypeFilterChange(opt.value)}
              className={`flex w-full items-center justify-between rounded px-2 py-1 text-sm transition-colors ${
                typeFilter === opt.value
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-text-muted hover:bg-surface2 hover:text-text"
              }`}
            >
              <span>{opt.label}</span>
              {typeCounts[opt.value] != null && (
                <span className="text-xs tabular-nums">
                  {typeCounts[opt.value]}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Sync filter */}
      <div>
        <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">
          Sync Status
        </h4>
        <div className="space-y-0.5">
          {SYNC_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              data-testid={`facet-sync-${opt.value}`}
              onClick={() => onSyncFilterChange(opt.value)}
              className={`flex w-full items-center justify-between rounded px-2 py-1 text-sm transition-colors ${
                syncFilter === opt.value
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-text-muted hover:bg-surface2 hover:text-text"
              }`}
            >
              <span>{opt.label}</span>
              {syncCounts[opt.value] != null && (
                <span className="text-xs tabular-nums">
                  {syncCounts[opt.value]}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tag filter */}
      {sortedTags.length > 0 && (
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted">
              Tags
            </h4>
            {tagFilter.length > 0 && (
              <button
                type="button"
                onClick={() =>
                  onTagMatchModeChange(
                    tagMatchMode === "any" ? "all" : "any"
                  )
                }
                className="text-xs text-primary hover:underline"
                data-testid="facet-tag-match-toggle"
              >
                {tagMatchMode === "any" ? "Any" : "All"}
              </button>
            )}
          </div>
          <div className="space-y-0.5">
            {visibleTags.map(([tag, count]) => (
              <button
                key={tag}
                type="button"
                data-testid={`facet-tag-${tag}`}
                onClick={() => toggleTag(tag)}
                className={`flex w-full items-center justify-between rounded px-2 py-1 text-sm transition-colors ${
                  tagFilter.includes(tag)
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-text-muted hover:bg-surface2 hover:text-text"
                }`}
              >
                <span className="truncate">{tag}</span>
                <span className="text-xs tabular-nums">{count}</span>
              </button>
            ))}
            {hiddenCount > 0 && !showAllTags && (
              <button
                type="button"
                onClick={() => setShowAllTags(true)}
                className="w-full px-2 py-1 text-xs text-primary hover:underline text-left"
                data-testid="facet-tags-show-more"
              >
                Show {hiddenCount} more...
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
