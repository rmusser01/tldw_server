import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import {
  Check,
  X,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  MessageSquare,
  Trash2,
  Search,
  BookOpen,
} from "lucide-react"
import type { WizardResultItem } from "./types"
import { shouldKeepOriginalFile } from "@/services/tldw/media-routing"
import { useIngestWizard } from "./IngestWizardContext"
import { classifyError } from "./ErrorClassification"
import type { ErrorCategory } from "./ErrorClassification"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type WizardResultsStepProps = {
  onClose: () => void
  onRetryItems?: (itemIds: string[]) => void
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
  onSearchKnowledge?: () => void
  onOpenWorkspace?: (item: WizardResultItem) => void
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a duration in milliseconds to a human-readable string ("Xs" or "M:SS"). */
function formatDuration(ms: number | undefined): string {
  if (ms == null || ms <= 0) return ""
  const totalSeconds = Math.round(ms / 1000)
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, "0")}`
}

/** Format elapsed seconds (from processing state) to "M:SS" or "Xs". */
function formatElapsed(seconds: number): string {
  if (seconds <= 0) return ""
  const rounded = Math.round(seconds)
  if (rounded < 60) return `${rounded}s`
  const m = Math.floor(rounded / 60)
  const s = rounded % 60
  return `${m}:${String(s).padStart(2, "0")}`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

type SuccessRowProps = {
  item: WizardResultItem
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
}

const SuccessRow: React.FC<SuccessRowProps> = React.memo(
  ({ item, qi, onOpenMedia, onDiscussInChat }) => {
    const label = item.title || item.fileName || item.url || item.id
    const duration = formatDuration(item.durationMs)

    const handleOpen = useCallback(() => onOpenMedia?.(item), [item, onOpenMedia])
    const handleChat = useCallback(() => onDiscussInChat?.(item), [item, onDiscussInChat])

    return (
      <div className="flex items-center gap-2 rounded-md px-3 py-2 hover:bg-surface2 transition-colors">
        <Check className="h-4 w-4 flex-shrink-0 text-green-500" aria-hidden="true" />
        <span className="min-w-0 flex-1 truncate text-sm text-text" title={label}>
          {label}
        </span>
        {duration && (
          <span className="flex-shrink-0 text-xs tabular-nums text-text-muted">
            {duration}
          </span>
        )}
        <div className="flex flex-shrink-0 items-center gap-1">
          {onOpenMedia && (
            <button
              type="button"
              onClick={handleOpen}
              className="rounded px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 transition-colors"
              aria-label={qi("wizard.results.openAria", "Open {{name}}", { name: label })}
            >
              <ExternalLink className="mr-0.5 inline h-3 w-3" aria-hidden="true" />
              {qi("wizard.results.open", "Open")}
            </button>
          )}
          {onDiscussInChat && (
            <button
              type="button"
              onClick={handleChat}
              className="rounded px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 transition-colors"
              aria-label={qi("wizard.results.chatAria", "Discuss {{name}} in chat", { name: label })}
            >
              <MessageSquare className="mr-0.5 inline h-3 w-3" aria-hidden="true" />
              {qi("wizard.results.chat", "Chat")}
            </button>
          )}
        </div>
      </div>
    )
  }
)
SuccessRow.displayName = "SuccessRow"

// ---------------------------------------------------------------------------

type SkippedRowProps = {
  item: WizardResultItem
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
}

const SkippedRow: React.FC<SkippedRowProps> = React.memo(
  ({ item, qi, onOpenMedia, onDiscussInChat }) => {
    const label = item.title || item.fileName || item.url || item.id

    const handleOpen = useCallback(() => onOpenMedia?.(item), [item, onOpenMedia])
    const handleChat = useCallback(() => onDiscussInChat?.(item), [item, onDiscussInChat])

    return (
      <div className="flex items-start gap-2 rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-500" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <span className="block truncate text-sm text-text" title={label}>
            {label}
          </span>
          <p className="mt-0.5 text-xs text-text-subtle">
            {item.message || qi(
              "wizard.results.skippedDefaultMessage",
              "This item already exists in your library. Use the \u2018Deep\u2019 preset to overwrite."
            )}
          </p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-1">
          {onOpenMedia && (
            <button
              type="button"
              onClick={handleOpen}
              className="rounded px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 transition-colors"
              aria-label={qi("wizard.results.openAria", "Open {{name}}", { name: label })}
            >
              <ExternalLink className="mr-0.5 inline h-3 w-3" aria-hidden="true" />
              {qi("wizard.results.open", "Open")}
            </button>
          )}
          {onDiscussInChat && (
            <button
              type="button"
              onClick={handleChat}
              className="rounded px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 transition-colors"
              aria-label={qi("wizard.results.chatAria", "Discuss {{name}} in chat", { name: label })}
            >
              <MessageSquare className="mr-0.5 inline h-3 w-3" aria-hidden="true" />
              {qi("wizard.results.chat", "Chat")}
            </button>
          )}
        </div>
      </div>
    )
  }
)
SkippedRow.displayName = "SkippedRow"

// ---------------------------------------------------------------------------

type ErrorRowProps = {
  item: WizardResultItem
  category: ErrorCategory
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onRetry?: (id: string) => void
  onRemove?: (id: string) => void
}

const ErrorRow: React.FC<ErrorRowProps> = React.memo(
  ({ item, category, qi, onRetry, onRemove }) => {
    const label = item.title || item.fileName || item.url || item.id

    const handleRetry = useCallback(() => onRetry?.(item.id), [item.id, onRetry])
    const handleRemove = useCallback(() => onRemove?.(item.id), [item.id, onRemove])

    return (
      <div className="rounded-md border border-danger/20 bg-danger/5 px-3 py-2">
        {/* Header row */}
        <div className="flex items-start gap-2">
          <X className="mt-0.5 h-4 w-4 flex-shrink-0 text-danger" aria-hidden="true" />
          <div className="min-w-0 flex-1">
            <span className="block truncate text-sm font-medium text-text" title={label}>
              {label}
            </span>
            {/* Plain-language explanation */}
            <p className="mt-1 text-xs text-text-subtle">
              {category.userMessage} {category.suggestion}
            </p>
          </div>
        </div>

        {/* Classification badge + actions */}
        <div className="mt-2 flex items-center justify-between">
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${category.badgeColor}`}
          >
            <AlertTriangle className="h-3 w-3" aria-hidden="true" />
            {category.badgeLabel}
          </span>

          <div className="flex items-center gap-1">
            {category.retryable && onRetry && (
              <button
                type="button"
                onClick={handleRetry}
                className="flex items-center gap-1 rounded px-2 py-1 text-xs text-primary hover:bg-primary/10 transition-colors"
                aria-label={qi("wizard.results.retryItemAria", "Retry {{name}}", { name: label })}
              >
                <RefreshCw className="h-3 w-3" aria-hidden="true" />
                {qi("wizard.results.retry", "Retry")}
              </button>
            )}
            {onRemove && (
              <button
                type="button"
                onClick={handleRemove}
                className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted hover:bg-danger/10 hover:text-danger transition-colors"
                aria-label={qi("wizard.results.removeItemAria", "Remove {{name}}", { name: label })}
              >
                <Trash2 className="h-3 w-3" aria-hidden="true" />
                {qi("wizard.results.remove", "Remove")}
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }
)
ErrorRow.displayName = "ErrorRow"

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const WizardResultsStep: React.FC<WizardResultsStepProps> = ({
  onClose,
  onRetryItems,
  onOpenMedia,
  onDiscussInChat,
  onSearchKnowledge,
  onOpenWorkspace,
}) => {
  const { t } = useTranslation(["option"])
  const { state, reset } = useIngestWizard()
  const { results, processingState } = state

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // -- Partition results into successes, skipped, and errors ----------------

  const { successes, skipped, errors } = useMemo(() => {
    const s: WizardResultItem[] = []
    const sk: WizardResultItem[] = []
    const e: WizardResultItem[] = []
    for (const item of results) {
      if (item.status === "error" || item.outcome === "failed") {
        e.push(item)
      } else if (item.outcome === "skipped") {
        sk.push(item)
      } else {
        s.push(item)
      }
    }
    return { successes: s, skipped: sk, errors: e }
  }, [results])

  // -- Classify each error --------------------------------------------------

  const errorCategories = useMemo(() => {
    const map = new Map<string, ErrorCategory>()
    for (const item of errors) {
      map.set(item.id, classifyError(item.error))
    }
    return map
  }, [errors])

  // -- Retryable error IDs --------------------------------------------------

  const retryableIds = useMemo(
    () => errors.filter((e) => errorCategories.get(e.id)?.retryable).map((e) => e.id),
    [errors, errorCategories]
  )

  // -- Callbacks ------------------------------------------------------------

  const handleRetryAll = useCallback(() => {
    if (retryableIds.length > 0) {
      onRetryItems?.(retryableIds)
    }
  }, [retryableIds, onRetryItems])

  const handleRetrySingle = useCallback(
    (id: string) => {
      onRetryItems?.([id])
    },
    [onRetryItems]
  )

  const handleRemoveSingle = useCallback(
    (_id: string) => {
      // Remove is a no-op placeholder; parent will handle via onRetryItems
      // or a future onRemoveItems callback.
    },
    []
  )

  const handleIngestMore = useCallback(() => {
    reset()
  }, [reset])

  // -- Elapsed time ---------------------------------------------------------

  const elapsedLabel = useMemo(
    () => formatElapsed(processingState.elapsed),
    [processingState.elapsed]
  )

  // -- Render ---------------------------------------------------------------

  return (
    <div className="flex h-full flex-col" data-testid="wizard-results-step">
      {/* Scrollable content area */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {/* Successes */}
        {successes.length > 0 && (
          <section aria-label={qi("wizard.results.completedSection", "Completed items")}>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
              <Check className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
              {qi("wizard.results.completedHeading", "Completed ({{count}})", {
                count: successes.length,
              })}
            </h3>
            <div className="space-y-0.5">
              {successes.map((item) => (
                <SuccessRow
                  key={item.id}
                  item={item}
                  qi={qi}
                  onOpenMedia={onOpenMedia}
                  onDiscussInChat={onDiscussInChat}
                />
              ))}
            </div>
          </section>
        )}

        {/* Next steps CTAs */}
        {successes.length > 0 && (onSearchKnowledge || onOpenWorkspace) && (
          <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3">
            <p className="mb-2 text-xs font-medium text-text-muted">
              {qi("wizard.results.nextSteps", "What's next?")}
            </p>
            <div className="flex flex-wrap gap-2">
              {onSearchKnowledge && (
                <button
                  type="button"
                  onClick={onSearchKnowledge}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.searchKnowledgeAria", "Search your ingested content in Knowledge QA")}
                >
                  <Search className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.searchKnowledge", "Search in Knowledge")}
                </button>
              )}
              {onOpenWorkspace && successes.some(s => s.persisted && shouldKeepOriginalFile(s.type)) && (
                <button
                  type="button"
                  onClick={() => {
                    const docItem = successes.find(s => s.persisted && shouldKeepOriginalFile(s.type))
                    if (docItem) onOpenWorkspace(docItem)
                  }}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.openWorkspaceAria", "Open document in Document Workspace")}
                >
                  <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.openWorkspace", "Open in Workspace")}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Skipped (duplicates) */}
        {skipped.length > 0 && (
          <section
            aria-label={qi("wizard.results.skippedSection", "Skipped items")}
            className={successes.length > 0 ? "mt-4" : ""}
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" aria-hidden="true" />
              {qi("wizard.results.skippedHeading", "Skipped ({{count}})", {
                count: skipped.length,
              })}
            </h3>
            <div className="space-y-1">
              {skipped.map((item) => (
                <SkippedRow
                  key={item.id}
                  item={item}
                  qi={qi}
                  onOpenMedia={onOpenMedia}
                  onDiscussInChat={onDiscussInChat}
                />
              ))}
            </div>
          </section>
        )}

        {/* Errors */}
        {errors.length > 0 && (
          <section
            aria-label={qi("wizard.results.errorsSection", "Error items")}
            className={successes.length > 0 || skipped.length > 0 ? "mt-4" : ""}
          >
            <div className="mb-2 flex items-center justify-between">
              <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-danger">
                <X className="h-3.5 w-3.5" aria-hidden="true" />
                {qi("wizard.results.errorsHeading", "Errors ({{count}})", {
                  count: errors.length,
                })}
              </h3>
              {retryableIds.length > 1 && onRetryItems && (
                <button
                  type="button"
                  onClick={handleRetryAll}
                  className="flex items-center gap-1 rounded-md bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                  aria-label={qi(
                    "wizard.results.retryAllAria",
                    "Retry all {{count}} retryable errors",
                    { count: retryableIds.length }
                  )}
                >
                  <RefreshCw className="h-3 w-3" aria-hidden="true" />
                  {qi("wizard.results.retryAll", "Retry All ({{count}})", {
                    count: retryableIds.length,
                  })}
                </button>
              )}
            </div>
            <div className="space-y-2">
              {errors.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error)}
                  qi={qi}
                  onRetry={onRetryItems ? handleRetrySingle : undefined}
                  onRemove={handleRemoveSingle}
                />
              ))}
            </div>
          </section>
        )}

        {/* Empty state */}
        {results.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center text-text-muted">
            <AlertTriangle className="mb-2 h-8 w-8 opacity-40" aria-hidden="true" />
            <p className="text-sm">
              {qi("wizard.results.noResults", "No results to display.")}
            </p>
          </div>
        )}
      </div>

      {/* Footer summary + actions */}
      {results.length > 0 && (
        <div className="border-t border-border px-4 py-3">
          {/* Summary line */}
          <p className="mb-3 text-center text-xs text-text-muted">
            {skipped.length > 0
              ? qi(
                  "wizard.results.summaryWithSkipped",
                  "Total: {{success}} succeeded, {{skipped}} skipped, {{failed}} failed",
                  {
                    success: successes.length,
                    skipped: skipped.length,
                    failed: errors.length,
                  }
                )
              : qi(
                  "wizard.results.summary",
                  "Total: {{success}} succeeded, {{failed}} failed",
                  { success: successes.length, failed: errors.length }
                )}
            {elapsedLabel && (
              <>
                {" \u00b7 "}
                {qi("wizard.results.elapsed", "{{time}} elapsed", { time: elapsedLabel })}
              </>
            )}
          </p>

          {/* Action buttons */}
          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={handleIngestMore}
              className="rounded-md border border-border bg-surface px-4 py-2 text-sm font-medium text-text hover:bg-surface2 transition-colors"
              aria-label={qi("wizard.results.ingestMoreAria", "Start a new ingest")}
            >
              {qi("wizard.results.ingestMore", "Ingest More")}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              aria-label={qi("wizard.results.doneAria", "Close the ingest wizard")}
            >
              {qi("wizard.results.done", "Done")}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

WizardResultsStep.displayName = "WizardResultsStep"

export default WizardResultsStep
