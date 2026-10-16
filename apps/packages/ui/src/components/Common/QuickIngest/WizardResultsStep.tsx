import React, { useCallback, useMemo, useState } from "react"
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
  Download,
} from "lucide-react"
import { getQueueItemExclusionReason, getEligibleQueueItems } from "./queue-items"
import type { WizardResultItem } from "./types"
import { shouldKeepOriginalFile } from "@/services/tldw/media-routing"
import {
  buildConferenceFailedResultExportText,
  buildConferenceRetryRequestItems,
  type ConferenceRetryRequestItem,
} from "@/services/tldw/conference-collections"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { useIngestWizard } from "./IngestWizardContext"
import { classifyError, extractionFailureSources } from "./ErrorClassification"
import type { ErrorCategory } from "./ErrorClassification"
import {
  canOpenMedia,
  canRetryWizardResult,
  getSavedMediaIds,
  GENERIC_SKIPPED_MESSAGE,
  LIBRARY_DUPLICATE_SKIP_MESSAGE,
  LOCAL_QUEUE_DUPLICATE_SKIP_MESSAGE,
  resolveSkippedResultReason,
} from "./result-actions"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type WizardResultsStepProps = {
  onClose: () => void
  onIngestMore?: () => void
  onRetryItems?: (
    itemIds: string[],
    retryItems?: ConferenceRetryRequestItem[]
  ) => void
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
  onSearchKnowledge?: (mediaIds: number[]) => void
  onOpenWorkspace?: (item: WizardResultItem) => void
  onOpenCollection?: (collectionId: string) => void
  onReviewSavedItems?: (ids: Array<string | number>) => void | Promise<void>
  onCorrectItems?: (ids: string[]) => void
  onReattach?: (id: string, file: File) => void
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

type ResultGroups = {
  successes: WizardResultItem[]
  warnings: WizardResultItem[]
  skippedExisting: WizardResultItem[]
  submitFailed: WizardResultItem[]
  failedProcessing: WizardResultItem[]
  cancelled: WizardResultItem[]
}

function groupResultItems(results: WizardResultItem[]): ResultGroups {
  const groups: ResultGroups = {
    successes: [],
    warnings: [],
    skippedExisting: [],
    submitFailed: [],
    failedProcessing: [],
    cancelled: [],
  }

  for (const item of results) {
    if (item.outcome === "skipped") {
      groups.skippedExisting.push(item)
    } else if (item.outcome === "submit_failed") {
      groups.submitFailed.push(item)
    } else if (item.outcome === "cancelled") {
      groups.cancelled.push(item)
    } else if (item.status === "error" || item.outcome === "failed") {
      groups.failedProcessing.push(item)
    } else if (item.warning) {
      groups.warnings.push(item)
    } else {
      groups.successes.push(item)
    }
  }

  return groups
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
    const showOpenMedia = Boolean(onOpenMedia) && canOpenMedia(item)

    const handleOpen = useCallback(() => onOpenMedia?.(item), [item, onOpenMedia])
    const handleChat = useCallback(() => onDiscussInChat?.(item), [item, onDiscussInChat])

    return (
      <div className="flex items-center gap-2 rounded-md px-3 py-2 hover:bg-surface2 transition-colors">
        {item.warning
          ? <AlertTriangle className="h-4 w-4 flex-shrink-0 text-amber-500" aria-hidden="true" />
          : <Check className="h-4 w-4 flex-shrink-0 text-green-500" aria-hidden="true" />}
        <div className="min-w-0 flex-1 text-sm text-text">
          <div className="truncate" title={label}>{label}</div>
          <p className="text-xs text-text-muted">{getSavedMediaIds([item]).length
            ? qi("wizard.results.savedState", "Saved")
            : qi("wizard.results.extractedState", "Extracted content; not saved")}
            {item.data && typeof item.data === "object" && "analysis" in item.data && typeof item.data.analysis === "string" && item.data.analysis.trim()
              ? ` · ${qi("wizard.results.analysisAvailable", "Analysis available")}` : ""}</p>
          {getSavedMediaIds([item]).length > 0 && <p className="text-xs text-text-muted">{qi("wizard.results.knowledgeUnconfirmed", "Knowledge readiness unconfirmed")}</p>}
          {item.warning && <p className="whitespace-pre-line break-words text-xs text-warn">{item.warning}</p>}
        </div>
        {duration && (
          <span className="flex-shrink-0 text-xs tabular-nums text-text-muted">
            {duration}
          </span>
        )}
        <div className="flex flex-shrink-0 items-center gap-1">
          {showOpenMedia && (
            <button
              type="button"
              onClick={handleOpen}
              className="rounded px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 transition-colors"
              aria-label={qi("wizard.results.openAria", "Open {{name}} in Media", { name: label })}
            >
              <ExternalLink className="mr-0.5 inline h-3 w-3" aria-hidden="true" />
              {qi("wizard.results.open", "Open in Media")}
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
    const showOpenMedia = Boolean(onOpenMedia) && canOpenMedia(item)
    const skippedReason = resolveSkippedResultReason(item)
    const skippedMessage =
      skippedReason === "local-queue-duplicate"
        ? qi("wizard.results.skippedAlreadyQueued", LOCAL_QUEUE_DUPLICATE_SKIP_MESSAGE)
        : skippedReason === "library-duplicate"
          ? qi("wizard.results.skippedAlreadyInLibrary", LIBRARY_DUPLICATE_SKIP_MESSAGE)
          : item.message || qi("wizard.results.skippedDefaultMessage", GENERIC_SKIPPED_MESSAGE)

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
            {skippedMessage}
          </p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-1">
          {showOpenMedia && (
            <button
              type="button"
              onClick={handleOpen}
              className="rounded px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 transition-colors"
              aria-label={qi("wizard.results.openAria", "Open {{name}} in Media", { name: label })}
            >
              <ExternalLink className="mr-0.5 inline h-3 w-3" aria-hidden="true" />
              {qi("wizard.results.open", "Open in Media")}
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
  onRetry?: (item: WizardResultItem) => void
  onRemove?: (id: string) => void
  onCorrect?: () => void
  onReattach?: (file: File) => void
  reattachError?: string
}

const MAX_LISTED_FAILED_SOURCES = 5

const ErrorRow: React.FC<ErrorRowProps> = React.memo(
  ({ item, category, qi, onRetry, onRemove, onCorrect, onReattach, reattachError }) => {
    const label = item.title || item.fileName || item.url || item.id
    const failedSources = extractionFailureSources(item.data)
    const hiddenSourceCount = failedSources.length - MAX_LISTED_FAILED_SOURCES

    const handleRetry = useCallback(() => onRetry?.(item), [item, onRetry])
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
            {failedSources.length > 0 && (
              <p className="mt-1 break-all text-xs text-text-subtle">
                {qi("wizard.results.failedSources", "Failed: {{sources}}", {
                  sources: failedSources.slice(0, MAX_LISTED_FAILED_SOURCES).join(", "),
                })}
                {hiddenSourceCount > 0 &&
                  ` ${qi("wizard.results.failedSourcesMore", "(+{{count}} more)", { count: hiddenSourceCount })}`}
              </p>
            )}
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

          <div className="flex flex-wrap items-center gap-2">
            {onReattach && <label className="text-xs text-text">
              {qi("wizard.results.reattach", "Reattach {{name}}", { name: label })}
              <input type="file" aria-label={qi("wizard.results.reattach", "Reattach {{name}}", { name: label })}
                className="mt-1 block max-w-full text-xs" onChange={event => { const file = event.target.files?.[0]; if (file) onReattach(file); event.target.value = "" }} />
              {reattachError && <span role="alert" className="block text-danger">{reattachError}</span>}
            </label>}
            {onCorrect && !onRetry && !onReattach && <button type="button" onClick={onCorrect}
              className="rounded px-2 py-1 text-xs text-primary hover:bg-primary/10"
              aria-label={qi("wizard.results.correctItem", "Correct settings for {{name}}", { name: label })}>
              {qi("wizard.results.correct", "Correct settings")}
            </button>}
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
  onIngestMore,
  onRetryItems,
  onOpenMedia,
  onDiscussInChat,
  onSearchKnowledge,
  onOpenWorkspace,
  onOpenCollection,
  onReviewSavedItems,
  onCorrectItems,
  onReattach,
}) => {
  const { t } = useTranslation(["option"])
  const { state, reset } = useIngestWizard()
  const { results, processingState } = state
  const queueItems = useMemo(() => state.queueItems ?? [], [state.queueItems]);
  const eligibleIds = new Set(getEligibleQueueItems(queueItems).map(item => item.id))
  const savedIds = getSavedMediaIds(results.filter(item => !queueItems.some(source => source.id === item.id) || eligibleIds.has(item.id) || queueItems.some(source => source.id === item.id && source.kind === "file" && !source.file)))
  const excludedItems = queueItems.filter(item => getQueueItemExclusionReason(item, queueItems) && !results.some(result => result.id === item.id))
  const missingFileIds = new Set(queueItems.filter(item => !item.url && !item.file).map(item => item.id))
  const tracking = useQuickIngestSessionStore((store) => store.session?.tracking)
  const { capabilities } = useServerCapabilities()
  const [exportNotice, setExportNotice] = useState<string | null>(null)

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // -- Partition results into conference-friendly outcome groups ------------

  const {
    successes,
    warnings,
    skippedExisting,
    submitFailed,
    failedProcessing,
    cancelled,
  } = useMemo(() => groupResultItems(results), [results])
  const savedItems = useMemo(() => [...successes, ...warnings], [successes, warnings])

  const failures = useMemo(
    () => [...submitFailed, ...failedProcessing, ...cancelled],
    [submitFailed, failedProcessing, cancelled]
  )

  const collectionId = tracking?.collectionId
  const hasDurableCollection =
    Boolean(collectionId) && tracking?.durableMode === "durable_collection"
  const savedMediaIds = getSavedMediaIds(savedItems)
    .filter(id => savedIds.includes(id))
    .map((id) => Number(id))
    .filter((id) => Number.isSafeInteger(id) && id > 0)
  const collectionMediaIds = savedIds
    .map((id) => Number(id))
    .filter((id) => Number.isSafeInteger(id) && id > 0)
  const canAskCollection = hasDurableCollection && collectionMediaIds.length > 0 && Boolean(onSearchKnowledge) && Boolean(capabilities?.hasKnowledgeQaMediaScope)
  const showGenericSearch =
    savedMediaIds.length > 0 && Boolean(onSearchKnowledge)
  const hasWorkspaceOpenTarget =
    Boolean(onOpenWorkspace) &&
    savedItems.some((item) => item.persisted && shouldKeepOriginalFile(item.type))
  const showCollectionOpen =
    hasDurableCollection && Boolean(onOpenCollection) && Boolean(collectionId)
  const showNextSteps =
    Boolean(savedIds.length && onReviewSavedItems) || showGenericSearch || hasWorkspaceOpenTarget || showCollectionOpen || canAskCollection

  // -- Classify each error --------------------------------------------------

  const errorCategories = useMemo(() => {
    const map = new Map<string, ErrorCategory>()
    for (const item of failures) {
      const category = classifyError(item.error, item.data)
      map.set(item.id, item.outcome === "cancelled" ? {
        ...category,
        retryable: false,
        badgeLabel: qi("wizard.results.cancelledBadge", "Cancelled"),
        userMessage: qi("wizard.results.cancelledMessage", "This item was cancelled."),
        suggestion: qi("wizard.results.cancelledSuggestion", "Review the input before starting another run.")
      } : category)
    }
    return map
  }, [failures, qi])

  // -- Retryable error IDs --------------------------------------------------

  const conferenceRetryRequests = useMemo(
    () => (hasDurableCollection ? buildConferenceRetryRequestItems(failures.filter(item => canRetryWizardResult(item, queueItems))) : []),
    [failures, hasDurableCollection, queueItems]
  )

  const conferenceRetryRequestsByResultId = useMemo(
    () =>
      new Map(
        conferenceRetryRequests.map((request) => [request.resultId, request])
      ),
    [conferenceRetryRequests]
  )

  const retryableIds = useMemo(
    () =>
      hasDurableCollection
        ? conferenceRetryRequests.map((request) => request.collectionItemId)
        : failures
            .filter((e) => canRetryWizardResult(e, queueItems))
            .map((e) => e.id),
    [conferenceRetryRequests, failures, hasDurableCollection, queueItems]
  )

  // -- Callbacks ------------------------------------------------------------

  const handleRetryAll = useCallback(() => {
    if (retryableIds.length > 0) {
      onRetryItems?.(
        retryableIds,
        hasDurableCollection ? conferenceRetryRequests : undefined
      )
    }
  }, [conferenceRetryRequests, hasDurableCollection, retryableIds, onRetryItems])

  const handleOpenCollection = useCallback(() => {
    if (collectionId) {
      onOpenCollection?.(collectionId)
    }
  }, [collectionId, onOpenCollection])

  const handleExportFailedItems = useCallback(async () => {
    const text = buildConferenceFailedResultExportText(failures)
    if (!text) return
    try {
      if (
        typeof navigator === "undefined" ||
        typeof navigator.clipboard?.writeText !== "function"
      ) {
        throw new Error("Clipboard unavailable")
      }
      await navigator.clipboard.writeText(text)
      setExportNotice(qi("wizard.results.failedExportCopied", "Failed item list copied."))
    } catch {
      if (typeof document !== "undefined" && typeof URL !== "undefined") {
        const blob = new Blob([text], { type: "text/plain" })
        const url = URL.createObjectURL(blob)
        const anchor = document.createElement("a")
        anchor.href = url
        anchor.download = "quick-ingest-failed-conference-items.txt"
        anchor.click()
        URL.revokeObjectURL(url)
        setExportNotice(qi("wizard.results.failedExportDownloaded", "Failed item list downloaded."))
      } else {
        setExportNotice(qi("wizard.results.failedExportUnavailable", "Failed item list could not be exported."))
      }
    }
  }, [failures, qi])

  const handleRetrySingle = useCallback(
    (item: WizardResultItem) => {
      if (hasDurableCollection) {
        const retryRequest = conferenceRetryRequestsByResultId.get(item.id)
        if (retryRequest) {
          onRetryItems?.([retryRequest.collectionItemId], [retryRequest])
        }
        return
      }
      onRetryItems?.([item.id])
    },
    [conferenceRetryRequestsByResultId, hasDurableCollection, onRetryItems]
  )

  const getRetryHandlerForItem = useCallback(
    (item: WizardResultItem) => {
      if (!onRetryItems || !canRetryWizardResult(item, queueItems)) return undefined
      if (!hasDurableCollection) return handleRetrySingle
      return conferenceRetryRequestsByResultId.has(item.id)
        ? handleRetrySingle
        : undefined
    },
    [
      conferenceRetryRequestsByResultId,
      handleRetrySingle,
      hasDurableCollection,
      onRetryItems,
      queueItems,
    ]
  )

  const handleIngestMore = useCallback(() => {
    if (onIngestMore) {
      onIngestMore()
      return
    }
    reset()
  }, [onIngestMore, reset])

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
        <p role="status" className="mb-3 text-sm text-text">
          {qi("wizard.results.batchAccounting", "{{added}} added · {{excluded}} excluded/skipped · {{succeeded}} succeeded ({{saved}} saved) · {{failed}} failed · {{cancelled}} cancelled", {
            added: new Set([...queueItems.map(item => item.id), ...results.map(item => item.id)]).size,
            excluded: excludedItems.length + skippedExisting.length,
            succeeded: savedItems.length, saved: savedIds.length,
            failed: submitFailed.length + failedProcessing.length, cancelled: cancelled.length,
          })}
        </p>
        {excludedItems.length > 0 && <section aria-label={qi("wizard.results.excludedSection", "Excluded inputs")} className="mb-4 space-y-2">
          {excludedItems.map(item => <div key={item.id} className="text-sm text-text">
            <p className="break-words">{item.fileName || item.url || item.id}</p>
            <p className="text-xs text-text-muted">{getQueueItemExclusionReason(item, queueItems) === "invalid"
              ? item.validation.errors?.join("; ") || qi("wizard.results.invalidExcluded", "Invalid input. Correct it before processing.")
              : getQueueItemExclusionReason(item, queueItems) === "unselected"
                ? qi("wizard.results.unselectedExcluded", "Not selected for this run.")
                : qi("wizard.results.skippedAlreadyQueued", LOCAL_QUEUE_DUPLICATE_SKIP_MESSAGE)}</p>
          </div>)}
        </section>}
        {/* Successes */}
        {successes.length > 0 && (
          <section aria-label={qi("wizard.results.completedSection", "Completed items")}>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
              <Check className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
              {qi("wizard.results.succeededHeading", "Succeeded ({{count}})", {
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

        {warnings.length > 0 && (
          <section aria-label={warnings.every(item => getSavedMediaIds([item]).length) ? qi("wizard.results.warningSection", "Items saved with warnings") : qi("wizard.results.unsavedWarnings", "Completed with warnings")} className={successes.length > 0 ? "mt-4" : ""}>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" aria-hidden="true" />
              {warnings.every(item => getSavedMediaIds([item]).length) ? qi("wizard.results.warningHeading", "Saved with warnings ({{count}})", { count: warnings.length }) : qi("wizard.results.unsavedWarningsHeading", "Completed with warnings ({{count}})", { count: warnings.length })}
            </h3>
            <div className="space-y-0.5">
              {warnings.map((item) => (
                <SuccessRow key={item.id} item={item} qi={qi} onOpenMedia={onOpenMedia} onDiscussInChat={onDiscussInChat} />
              ))}
            </div>
          </section>
        )}

        {/* Next steps CTAs */}
        {showNextSteps && (
          <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3">
            <p className="mb-2 text-xs font-medium text-text-muted">
              {qi("wizard.results.nextSteps", "What's next?")}
            </p>
            <div className="flex flex-wrap gap-2">
              {savedIds.length > 0 && onReviewSavedItems && <button type="button" onClick={() => {
                void Promise.resolve(onReviewSavedItems(savedIds)).catch(() => setExportNotice(qi("wizard.results.savedReviewFailed", "Unable to open saved items. Try again.")))
              }} className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text hover:bg-surface2">
                {qi("wizard.results.reviewSaved", savedIds.length === 1 ? "Review this {{count}} saved item" : "Review these {{count}} saved items", { count: savedIds.length })}
              </button>}
              {showCollectionOpen && collectionId && (
                <button
                  type="button"
                  onClick={handleOpenCollection}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi(
                    "wizard.results.openCollectionAria",
                    "Open collection {{collectionId}}",
                    { collectionId }
                  )}
                >
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.openCollection", "Open collection")}
                </button>
              )}
              {canAskCollection && (
                <button
                  type="button"
                  onClick={() => onSearchKnowledge?.(collectionMediaIds)}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.askCollectionAria", "Ask this collection")}
                >
                  <MessageSquare className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.askCollection", "Ask this collection")}
                </button>
              )}
              {showGenericSearch && onSearchKnowledge && (
                <button
                  type="button"
                  onClick={() => onSearchKnowledge(savedMediaIds)}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi(
                    "wizard.results.askAddedItemsAria",
                    "Ask added items",
                  )}
                >
                  <Search className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.askAddedItems", "Ask added items")}
                </button>
              )}
              {hasWorkspaceOpenTarget && onOpenWorkspace && (
                <button
                  type="button"
                  onClick={() => {
                    const docItem = savedItems.find(s => s.persisted && shouldKeepOriginalFile(s.type))
                    if (docItem) onOpenWorkspace(docItem)
                  }}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.openDocumentWorkspaceAria", "Open in Document Workspace")}
                >
                  <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.openDocumentWorkspace", "Open in Document Workspace")}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Skipped (duplicates) */}
        {skippedExisting.length > 0 && (
          <section
            aria-label={qi("wizard.results.skippedSection", "Skipped items")}
            className={savedItems.length > 0 ? "mt-4" : ""}
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" aria-hidden="true" />
              {qi("wizard.results.skippedExistingHeading", "Skipped existing ({{count}})", {
                count: skippedExisting.length,
              })}
            </h3>
            <div className="space-y-1">
              {skippedExisting.map((item) => (
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

        {/* Failure export/retry actions */}
        {failures.length > 0 && (
          <div
            className={
              savedItems.length > 0 || skippedExisting.length > 0
                ? "mt-4 flex flex-wrap items-center justify-between gap-2 rounded-md border border-danger/15 bg-danger/5 px-3 py-2"
                : "flex flex-wrap items-center justify-between gap-2 rounded-md border border-danger/15 bg-danger/5 px-3 py-2"
            }
          >
            <span className="text-xs font-medium text-danger">
              {qi("wizard.results.reviewFailedItems", "Review failed items")}
            </span>
            <div className="flex flex-wrap items-center gap-2">
              {exportNotice && (
                <span className="text-xs text-text-muted">{exportNotice}</span>
              )}
              <button
                type="button"
                onClick={() => {
                  void handleExportFailedItems()
                }}
                className="flex items-center gap-1 rounded-md border border-border bg-surface px-2.5 py-1 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                aria-label={qi("wizard.results.exportFailedListAria", "Export failed items list")}
              >
                <Download className="h-3 w-3" aria-hidden="true" />
                {qi("wizard.results.exportFailedList", "Export failed list")}
              </button>
              {retryableIds.length > 0 && onRetryItems && (
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
          </div>
        )}

        {/* Submission failures */}
        {submitFailed.length > 0 && (
          <section
            aria-label={qi("wizard.results.submitFailedSection", "Not submitted items")}
            className="mt-4"
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-danger">
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              {qi("wizard.results.submitFailedHeading", "Not submitted ({{count}})", {
                count: submitFailed.length,
              })}
            </h3>
            <div className="space-y-2">
              {submitFailed.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error, item.data)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                  onCorrect={onCorrectItems ? () => onCorrectItems([item.id]) : undefined}
                  onReattach={missingFileIds.has(item.id) && onReattach ? file => onReattach(item.id, file) : undefined}
                  reattachError={queueItems.find(source => source.id === item.id)?.validation.errors?.join("; ")}
                />
              ))}
            </div>
          </section>
        )}

        {/* Processing failures */}
        {failedProcessing.length > 0 && (
          <section
            aria-label={qi("wizard.results.errorsSection", "Error items")}
            className="mt-4"
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-danger">
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              {qi("wizard.results.failedProcessingHeading", "Failed during processing ({{count}})", {
                count: failedProcessing.length,
              })}
            </h3>
            <div className="space-y-2">
              {failedProcessing.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error, item.data)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                  onCorrect={onCorrectItems ? () => onCorrectItems([item.id]) : undefined}
                  onReattach={missingFileIds.has(item.id) && onReattach ? file => onReattach(item.id, file) : undefined}
                  reattachError={queueItems.find(source => source.id === item.id)?.validation.errors?.join("; ")}
                />
              ))}
            </div>
          </section>
        )}

        {/* Cancelled */}
        {cancelled.length > 0 && (
          <section
            aria-label={qi("wizard.results.cancelledSection", "Cancelled items")}
            className="mt-4"
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              {qi("wizard.results.cancelledHeading", "Cancelled ({{count}})", {
                count: cancelled.length,
              })}
            </h3>
            <div className="space-y-2">
              {cancelled.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error, item.data)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                  onCorrect={onCorrectItems ? () => onCorrectItems([item.id]) : undefined}
                  onReattach={missingFileIds.has(item.id) && onReattach ? file => onReattach(item.id, file) : undefined}
                  reattachError={queueItems.find(source => source.id === item.id)?.validation.errors?.join("; ")}
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
            {elapsedLabel && (
              <>
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
