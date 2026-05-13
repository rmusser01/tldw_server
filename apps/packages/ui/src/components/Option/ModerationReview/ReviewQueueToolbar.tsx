import React from "react"
import { RefreshCw, Search } from "lucide-react"

import { getDesignSystemState } from "@/design-system"
import type { ModerationReviewFilters } from "./hooks/useModerationReviewQueue"

type ReviewQueueToolbarProps = {
  filters: ModerationReviewFilters
  onFilterChange: <K extends keyof ModerationReviewFilters>(key: K, value: ModerationReviewFilters[K]) => void
  onRefresh: () => void
  loading?: boolean
  compact?: boolean
}

const inputClass =
  "rounded-md border border-border bg-surface px-2.5 py-2 text-sm text-text outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
const BLOCKED_STATE_LABEL = getDesignSystemState("blocked").label

export const ReviewQueueToolbar: React.FC<ReviewQueueToolbarProps> = ({
  filters,
  onFilterChange,
  onRefresh,
  loading = false,
  compact = false
}) => {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className={`grid gap-3 ${compact ? "grid-cols-1" : "md:grid-cols-4 xl:grid-cols-8"}`}>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Status
          <select
            className={inputClass}
            value={filters.status}
            onChange={(event) => onFilterChange("status", event.target.value as ModerationReviewFilters["status"])}
          >
            <option value="">All statuses</option>
            <option value="needs_review">Needs review</option>
            <option value="approved">Approved</option>
            <option value="blocked">{BLOCKED_STATE_LABEL}</option>
            <option value="redacted">Redacted</option>
            <option value="dismissed">Dismissed</option>
            <option value="escalated">Escalated</option>
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Severity
          <select
            className={inputClass}
            value={filters.severity}
            onChange={(event) => onFilterChange("severity", event.target.value as ModerationReviewFilters["severity"])}
          >
            <option value="">Any severity</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Category
          <input
            className={inputClass}
            value={filters.category}
            onChange={(event) => onFilterChange("category", event.target.value)}
            placeholder="pii"
          />
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Source type
          <input
            className={inputClass}
            value={filters.source_type}
            onChange={(event) => onFilterChange("source_type", event.target.value)}
            placeholder="chat"
          />
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Source ID
          <input
            className={inputClass}
            value={filters.source_id}
            onChange={(event) => onFilterChange("source_id", event.target.value)}
            placeholder="conversation"
          />
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          User ID
          <input
            className={inputClass}
            value={filters.user_id}
            onChange={(event) => onFilterChange("user_id", event.target.value)}
            placeholder="user"
          />
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Sort
          <select
            className={inputClass}
            value={filters.sort}
            onChange={(event) => onFilterChange("sort", event.target.value as ModerationReviewFilters["sort"])}
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium text-text-muted">
          Search
          <span className="relative">
            <Search className="pointer-events-none absolute left-2 top-2.5 h-4 w-4 text-text-muted" aria-hidden="true" />
            <input
              className={`${inputClass} w-full pl-8`}
              value={filters.q}
              onChange={(event) => onFilterChange("q", event.target.value)}
              placeholder="Excerpt or source"
            />
          </span>
        </label>
      </div>
      <div className="mt-3 flex justify-end">
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text hover:bg-surface3 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} aria-hidden="true" />
          Refresh
        </button>
      </div>
    </div>
  )
}
