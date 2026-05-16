import React from "react"
import { useTranslation } from "react-i18next"
import { SearchX } from "lucide-react"
import { Link } from "react-router-dom"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { getRagSourceLabel, isRagSource } from "@/services/rag/sourceMetadata"
import type { RagSource } from "@/services/rag/unified-rag"
import type {
  KnowledgeSourceHealth,
  KnowledgeSourceHealthState,
  KnowledgeSourceStatus,
} from "../types"
import { getSourceHealthStatusLabel } from "../sourceHealth"

type NoResultsRecoveryProps = {
  onOpenQuickIngest: () => void
  onEnableWeb: () => void
  onShowNearestMatches: () => void
  webEnabled: boolean
  selectedSources?: RagSource[]
  sourceHealth?: KnowledgeSourceHealthState
  sourceStatus?: Record<string, KnowledgeSourceStatus>
  showNearestMatchesAvailable?: boolean
}

export function NoResultsRecovery({
  onOpenQuickIngest,
  onEnableWeb,
  onShowNearestMatches,
  webEnabled,
  selectedSources = [],
  sourceHealth,
  sourceStatus,
  showNearestMatchesAvailable = false,
}: NoResultsRecoveryProps) {
  const { t } = useTranslation("knowledge")
  const recentlyIngestedDocs = useQuickIngestStore(s => s.recentlyIngestedDocs)
  const hasRecentIngests = recentlyIngestedDocs.length > 0
  const sourceDiagnostics = React.useMemo(
    () =>
      Object.entries(sourceStatus ?? {}).map(([sourceId, status]) => ({
        sourceId,
        label: isRagSource(sourceId) ? getRagSourceLabel(sourceId) : sourceId,
        status,
      })),
    [sourceStatus]
  )
  const sourceReadiness = React.useMemo(() => {
    if (!sourceHealth || sourceHealth.error) return []
    const sourceIds =
      selectedSources.length > 0
        ? selectedSources
        : sourceHealth.sources.map((source) => source.sourceId)
    return sourceIds
      .map((sourceId) => sourceHealth.bySource[sourceId])
      .filter((source): source is KnowledgeSourceHealth => Boolean(source))
  }, [selectedSources, sourceHealth])

  return (
    <div className="rounded-xl border border-border bg-surface p-6">
      <div className="flex items-start gap-3">
        <SearchX className="mt-0.5 h-5 w-5 text-text-muted" />
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold">No results found</h2>
          <p className="mt-1 text-sm text-text-muted">
            Try broader keywords, check source readiness and search diagnostics, or enable web fallback for recovery.
          </p>
          <p className="mt-1 text-sm text-text-muted">
            Web fallback uses your configured server default provider. Queries stay on your
            tldw server unless you enable web fallback.
          </p>
          {hasRecentIngests && (
            <div className="mb-3 mt-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2">
              <p className="text-xs text-amber-700 dark:text-amber-400">
                {t("knowledge:noResults.indexingHint", "You recently ingested documents. If they don't appear in results yet, they may still be indexing. Try searching again in a moment.")}
              </p>
            </div>
          )}
          {sourceReadiness.length > 0 ? (
            <div className="mt-3 rounded-md border border-border/80 bg-surface2/50 px-3 py-2">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Source readiness
              </h3>
              <ul className="mt-1 space-y-1 text-xs text-text-muted">
                {sourceReadiness.map((health) => (
                  <li key={health.sourceId}>
                    {health.label}: {getSourceHealthStatusLabel(health).toLowerCase()}
                    {health.disabledReason
                      ? `, ${health.disabledReason.replaceAll("_", " ")}`
                      : ""}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {sourceDiagnostics.length > 0 ? (
            <div className="mt-3 rounded-md border border-border/80 bg-surface2/50 px-3 py-2">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Search diagnostics
              </h3>
              <ul className="mt-1 space-y-1 text-xs text-text-muted">
                {sourceDiagnostics.map(({ sourceId, label, status }) => (
                  <li key={sourceId}>
                    {label}: {status.status}
                    {status.count > 0 ? ` (${status.count} found)` : ""}
                    {status.reason ? `, ${status.reason.replaceAll("_", " ")}` : ""}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <ul className="mt-2 space-y-1 text-sm text-text-muted">
            <li>Try different keywords or fewer constraints.</li>
            <li>Broaden the question before adding details.</li>
            <li>Confirm your sources were ingested and indexed.</li>
          </ul>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onOpenQuickIngest}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Open Quick Ingest
            </button>
            <Link
              to="/sources"
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Open source page
            </Link>
            <button
              type="button"
              onClick={onEnableWeb}
              disabled={webEnabled}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-subtle disabled:opacity-60 disabled:cursor-not-allowed hover:bg-hover hover:text-text transition-colors"
            >
              {webEnabled ? "Web fallback enabled" : "Enable web fallback"}
            </button>
            {showNearestMatchesAvailable ? (
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
