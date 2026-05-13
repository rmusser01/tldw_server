import React from "react"
import { Link } from "react-router-dom"
import { AlertTriangle, CheckCircle2, ClipboardList, ShieldCheck, SlidersHorizontal } from "lucide-react"

import { useConnectionUxState } from "@/hooks/useConnectionState"
import { useServerOnline } from "@/hooks/useServerOnline"
import { MODERATION_RULES_PATH } from "@/routes/route-paths"
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

export const ModerationReviewShell: React.FC<ModerationReviewShellProps> = ({
  compact = false
}) => {
  const online = useServerOnline()
  const { uxState } = useConnectionUxState()
  const backendStatus = backendStatusCopy(online, uxState)
  const StatusIcon = backendStatus.tone === "ok" ? CheckCircle2 : AlertTriangle
  const queue = useModerationReviewQueue()
  const visibleTotal = queue.total ?? queue.items.length
  const selectedStatus = queue.selectedItem?.status || "None"

  return (
    <section
      className="space-y-5"
      data-testid="moderation-review-shell"
      aria-labelledby="moderation-review-title"
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
      />

      <ReviewStatePanels
        loading={queue.loading}
        error={queue.error}
        partial={queue.partial}
        warnings={queue.warnings}
        empty={!queue.loading && !queue.error && queue.items.length === 0 && !queue.selectedItem}
        onRetry={queue.refresh}
      />

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
                onSelect={(itemId) => void queue.selectItem(itemId)}
              />
            ) : (
              <div className="rounded-lg border border-border bg-surface2 p-4 text-sm text-text-muted">
                The active filters no longer include this selected item.
              </div>
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
              onUndo={queue.undoDecision}
            />
          </aside>
        </div>
      )}
    </section>
  )
}
