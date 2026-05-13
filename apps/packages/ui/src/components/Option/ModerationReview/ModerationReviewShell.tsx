import React from "react"
import { Link } from "react-router-dom"
import { AlertTriangle, CheckCircle2, ClipboardList, ShieldCheck, SlidersHorizontal } from "lucide-react"

import { useConnectionUxState } from "@/hooks/useConnectionState"
import { useServerOnline } from "@/hooks/useServerOnline"
import { MODERATION_RULES_PATH } from "@/routes/route-paths"
import { BulkDecisionBar } from "./BulkDecisionBar"
import { DecisionBar } from "./DecisionBar"
import { useModerationReviewQueue } from "./hooks/useModerationReviewQueue"
import { ReviewItemDetail } from "./ReviewItemDetail"
import { ReviewQueueList } from "./ReviewQueueList"
import { ReviewQueueToolbar } from "./ReviewQueueToolbar"
import { ReviewStatePanels } from "./ReviewStatePanels"

type ModerationReviewShellProps = {
  compact?: boolean
}

const backendStatusCopy = (online: boolean, uxState: string) => {
  if (online) {
    return {
      tone: "ok" as const,
      title: "Server reachable",
      description: "Review queue endpoints are available when this server includes the moderation review backend."
    }
  }
  if (uxState === "error_auth" || uxState === "configuring_auth") {
    return {
      tone: "warn" as const,
      title: "Credentials needed",
      description: "Connect credentials before moderation review data can load."
    }
  }
  if (uxState === "unconfigured" || uxState === "configuring_url") {
    return {
      tone: "warn" as const,
      title: "Server setup incomplete",
      description: "Finish setup before moderation review data can load."
    }
  }
  return {
    tone: "warn" as const,
    title: "Server unreachable",
    description: "Review queue data is unavailable until the tldw server responds."
  }
}

const shouldIgnoreShortcut = (target: EventTarget | null) => {
  const element = target as HTMLElement | null
  if (!element) {
    return false
  }
  const tagName = element.tagName.toLowerCase()
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    element.isContentEditable
  )
}

export const ModerationReviewShell: React.FC<ModerationReviewShellProps> = ({
  compact = false
}) => {
  const online = useServerOnline()
  const { uxState } = useConnectionUxState()
  const backendStatus = backendStatusCopy(online, uxState)
  const StatusIcon = backendStatus.tone === "ok" ? CheckCircle2 : AlertTriangle
  const queue = useModerationReviewQueue()
  const searchInputRef = React.useRef<HTMLInputElement | null>(null)
  const visibleTotal = queue.total ?? queue.items.length
  const selectedStatus = queue.selectedItem?.status || "None"
  const reviewComplete =
    !queue.loading &&
    !queue.error &&
    queue.filters.status === "needs_review" &&
    visibleTotal === 0 &&
    queue.items.length === 0

  const handleShortcut = React.useCallback(
    (event: React.KeyboardEvent<HTMLElement>) => {
      if (event.metaKey || event.ctrlKey || event.altKey) {
        return
      }
      if (shouldIgnoreShortcut(event.target)) {
        return
      }
      if (event.key === "n" || event.key === "ArrowDown") {
        event.preventDefault()
        void queue.selectRelative(1)
      } else if (event.key === "p" || event.key === "ArrowUp") {
        event.preventDefault()
        void queue.selectRelative(-1)
      } else if (event.key === "a") {
        event.preventDefault()
        void queue.decideSelected("approve")
      } else if (event.key === "d") {
        event.preventDefault()
        void queue.decideSelected("dismiss")
      } else if (event.key === "/") {
        event.preventDefault()
        searchInputRef.current?.focus()
      } else if (event.key === "r") {
        event.preventDefault()
        void queue.refresh()
      }
    },
    [queue]
  )

  return (
    <section
      className="space-y-5"
      data-testid="moderation-review-shell"
      aria-labelledby="moderation-review-title"
      tabIndex={0}
      onKeyDown={handleShortcut}
      title="Shortcuts: n next, p previous, a approve, d dismiss, / search, r refresh"
    >
      <div className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-5 shadow-sm sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-3xl">
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-border bg-surface2 px-3 py-1 text-xs font-medium text-text-muted">
            <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
            Review queue
          </div>
          <h1
            id="moderation-review-title"
            className="text-2xl font-semibold text-text"
          >
            Moderation Review
          </h1>
          <p className="mt-2 text-sm leading-6 text-text-muted">
            Review sanitized moderation outcomes, inspect policy context, and record decisions with an audit trail.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {compact && (
            <Link
              to="/moderation"
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text transition hover:bg-surface3"
            >
              <ClipboardList className="h-4 w-4" aria-hidden="true" />
              Open full review
            </Link>
          )}
          <Link
            to={MODERATION_RULES_PATH}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text transition hover:bg-surface3"
          >
            <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
            Open Content Rules
          </Link>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="text-sm font-medium text-text-muted">Matching items</div>
          <div className="mt-2 text-2xl font-semibold text-text">{queue.loading ? "--" : visibleTotal}</div>
          <div className="mt-1 text-xs text-text-muted">Current filter result count</div>
        </div>
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="text-sm font-medium text-text-muted">Selected status</div>
          <div className="mt-2 text-2xl font-semibold capitalize text-text">{selectedStatus.replace("_", " ")}</div>
          <div className="mt-1 text-xs text-text-muted">Updates after each decision</div>
        </div>
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="flex items-start gap-3">
            <StatusIcon
              className={
                backendStatus.tone === "ok"
                  ? "mt-0.5 h-4 w-4 text-green-600"
                  : "mt-0.5 h-4 w-4 text-yellow-600"
              }
              aria-hidden="true"
            />
            <div>
              <div className="text-sm font-semibold text-text">{backendStatus.title}</div>
              <p className="mt-1 text-xs text-text-muted">{backendStatus.description}</p>
            </div>
          </div>
        </div>
      </div>

      <ReviewQueueToolbar
        filters={queue.filters}
        onFilterChange={queue.updateFilter}
        onRefresh={queue.refresh}
        loading={queue.loading}
        compact={compact}
        searchInputRef={searchInputRef}
        filterPresets={queue.filterPresets}
        onSavePreset={queue.saveFilterPreset}
        onApplyPreset={queue.applyFilterPreset}
        onDeletePreset={queue.deleteFilterPreset}
      />

      <ReviewStatePanels
        loading={queue.loading}
        error={queue.error}
        partial={queue.partial}
        warnings={queue.warnings}
        empty={!reviewComplete && !queue.loading && !queue.error && queue.items.length === 0 && !queue.selectedItem}
        onRetry={queue.refresh}
      />

      {reviewComplete && (
        <div className="rounded-lg border border-green-200 bg-green-50 p-4 text-sm text-green-900 dark:border-green-900/50 dark:bg-green-950/30 dark:text-green-100">
          <div className="font-semibold">Review complete</div>
          <p className="mt-1">No items currently need review under the active filters.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <a
              href="#moderation-review-audit"
              className="rounded-md border border-green-300 bg-white px-3 py-2 font-medium text-green-900 hover:bg-green-100 dark:border-green-900/50 dark:bg-green-950/40 dark:text-green-100"
            >
              Review audit
            </a>
            <Link
              to={MODERATION_RULES_PATH}
              className="rounded-md border border-green-300 bg-white px-3 py-2 font-medium text-green-900 hover:bg-green-100 dark:border-green-900/50 dark:bg-green-950/40 dark:text-green-100"
            >
              Content rules
            </Link>
          </div>
        </div>
      )}

      {!queue.loading && !queue.error && (queue.items.length > 0 || queue.selectedItem) && (
        <div className={`grid gap-4 ${compact ? "grid-cols-1" : "xl:grid-cols-[minmax(0,1.15fr)_minmax(340px,0.85fr)]"}`}>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-text">
              <ClipboardList className="h-4 w-4" aria-hidden="true" />
              Review worklist
            </div>
            {queue.items.length > 0 ? (
              <ReviewQueueList
                items={queue.items}
                selectedItemId={queue.selectedItemId}
                selectedForBulkIds={queue.selectedItemIds}
                onSelect={(itemId) => void queue.selectItem(itemId)}
                onToggleSelected={queue.toggleSelected}
              />
            ) : (
              <div className="rounded-lg border border-border bg-surface2 p-4 text-sm text-text-muted">
                The active filters no longer include this selected item.
              </div>
            )}
            {queue.selectedItemIds.length > 0 && (
              <BulkDecisionBar
                selectedCount={queue.selectedItemIds.length}
                deciding={queue.bulkDeciding}
                result={queue.bulkResult}
                onBulkDecision={queue.bulkDecideSelected}
                onClearSelection={queue.clearSelection}
              />
            )}
            {queue.nextCursor && (
              <button
                type="button"
                onClick={() => void queue.loadNextPage()}
                className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text hover:bg-surface3"
              >
                Load next page
              </button>
            )}
          </div>

          <aside className="space-y-3">
            <ReviewItemDetail item={queue.selectedItem} loading={queue.detailLoading} />
            <DecisionBar
              disabled={!queue.selectedItem}
              deciding={queue.deciding}
              onDecision={queue.decideSelected}
              undoToken={queue.undo?.token || null}
              undoExpiresAt={queue.undo?.expiresAt || null}
              onUndo={queue.undoDecision}
            />
          </aside>
        </div>
      )}
    </section>
  )
}
