import React from "react"
import { useTranslation } from "react-i18next"
import { SearchX } from "lucide-react"
import { useQuickIngestStore } from "@/store/quick-ingest"

type NoResultsRecoveryProps = {
  onBroadenScope: () => void
  onEnableWeb: () => void
  onShowNearestMatches: () => void
  webEnabled: boolean
  webAvailable?: boolean
  hasNearestMatches?: boolean
}

export function NoResultsRecovery({
  onBroadenScope,
  onEnableWeb,
  onShowNearestMatches,
  webEnabled,
  webAvailable = true,
  hasNearestMatches = false,
}: NoResultsRecoveryProps) {
  const { t } = useTranslation("knowledge")
  const recentlyIngestedDocs = useQuickIngestStore(s => s.recentlyIngestedDocs)
  const hasRecentIngests = recentlyIngestedDocs.length > 0

  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <div className="flex items-start gap-3">
        <SearchX className="mt-0.5 h-5 w-5 text-text-muted" />
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold">No results found</h2>
          <p className="mt-1 text-sm text-text-muted">
            Try broader sources, adjust the question, or use recovery options available for this server.
          </p>
          {hasRecentIngests && (
            <div className="mb-3 mt-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2">
              <p className="text-xs text-amber-700 dark:text-amber-400">
                {t("knowledge:noResults.indexingHint", "You recently ingested documents. If they don't appear in results yet, they may still be indexing. Try searching again in a moment.")}
              </p>
            </div>
          )}
          <ul className="mt-2 space-y-1 text-sm text-text-muted">
            <li>Try different keywords or fewer constraints.</li>
            <li>Broaden the question before adding details.</li>
            <li>Confirm your sources were ingested and indexed.</li>
          </ul>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onBroadenScope}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Broaden source scope
            </button>
            {webAvailable && !webEnabled ? (
              <button
                type="button"
                onClick={onEnableWeb}
                className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-subtle hover:bg-hover hover:text-text transition-colors"
              >
                Enable web search
              </button>
            ) : null}
            {webAvailable && webEnabled ? (
              <span className="rounded-md border border-info/30 bg-info/10 px-3 py-1.5 text-sm text-info">
                Web search enabled
              </span>
            ) : null}
            {!webAvailable ? (
              <span className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-muted">
                Web search unavailable on this server
              </span>
            ) : null}
            {hasNearestMatches ? (
              <button
                type="button"
                onClick={onShowNearestMatches}
                className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-subtle hover:bg-hover hover:text-text transition-colors"
              >
                Show nearest matches
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}
