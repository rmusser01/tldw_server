import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useVirtualizer } from "@tanstack/react-virtual"
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
import { classifyError } from "./ErrorClassification"
import type { ErrorCategory } from "./ErrorClassification"
import {
  canOpenMedia,
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
  onSearchKnowledge?: () => void
  onOpenWorkspace?: (item: WizardResultItem) => void
  onOpenCollection?: (collectionId: string) => void
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
  completed: WizardResultItem[]
  includedExisting: WizardResultItem[]
  metadataUpdated: WizardResultItem[]
  skippedExisting: WizardResultItem[]
  submitFailed: WizardResultItem[]
  processingFailed: WizardResultItem[]
  metadataUpdateFailed: WizardResultItem[]
  cancelled: WizardResultItem[]
}

type TerminalOutcome = NonNullable<WizardResultItem["terminalOutcome"]>
type ResultFilter = "all" | TerminalOutcome

function terminalOutcomeForResult(item: WizardResultItem): TerminalOutcome {
  if (item.terminalOutcome) return item.terminalOutcome
  if (item.outcome === "skipped") return "skipped_existing"
  if (item.outcome === "submit_failed") return "submit_failed"
  if (item.outcome === "cancelled") return "cancelled"
  if (item.status === "error" || item.outcome === "failed") {
    return "processing_failed"
  }
  return "completed"
}

function groupResultItems(results: WizardResultItem[]): ResultGroups {
  const groups: ResultGroups = {
    completed: [],
    includedExisting: [],
    metadataUpdated: [],
    skippedExisting: [],
    submitFailed: [],
    processingFailed: [],
    metadataUpdateFailed: [],
    cancelled: [],
  }

  for (const item of results) {
    const outcome = terminalOutcomeForResult(item)
    if (outcome === "completed") groups.completed.push(item)
    else if (outcome === "included_existing") groups.includedExisting.push(item)
    else if (outcome === "metadata_updated") groups.metadataUpdated.push(item)
    else if (outcome === "skipped_existing") groups.skippedExisting.push(item)
    else if (outcome === "submit_failed") groups.submitFailed.push(item)
    else if (outcome === "processing_failed") groups.processingFailed.push(item)
    else if (outcome === "metadata_update_failed") groups.metadataUpdateFailed.push(item)
    else groups.cancelled.push(item)
  }

  return groups
}

function hasReadyMedia(item: WizardResultItem): boolean {
  return item.mediaId != null
}

function terminalOutcomeLabel(
  outcome: TerminalOutcome,
  qi: (key: string, defaultValue: string) => string
): string {
  switch (outcome) {
    case "included_existing":
      return qi("wizard.results.includedExisting", "Included existing")
    case "metadata_updated":
      return qi("wizard.results.metadataUpdated", "Metadata updated")
    case "skipped_existing":
      return qi("wizard.results.skippedExisting", "Skipped existing")
    case "submit_failed":
      return qi("wizard.results.notSubmitted", "Not submitted")
    case "processing_failed":
      return qi("wizard.results.processingFailed", "Failed during processing")
    case "metadata_update_failed":
      return qi("wizard.results.metadataUpdateFailed", "Metadata update failed")
    case "cancelled":
      return qi("wizard.results.cancelled", "Cancelled")
    default:
      return qi("wizard.results.completed", "Completed")
  }
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
}

const ErrorRow: React.FC<ErrorRowProps> = React.memo(
  ({ item, category, qi, onRetry, onRemove }) => {
    const label = item.title || item.fileName || item.url || item.id
    const retryable = item.retryable ?? category.retryable

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
            {retryable && onRetry && (
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
}) => {
  const { t } = useTranslation(["option"])
  const { state, reset } = useIngestWizard()
  const { results, processingState } = state
  const tracking = useQuickIngestSessionStore((store) => store.session?.tracking)
  const { capabilities } = useServerCapabilities()
  const [exportNotice, setExportNotice] = useState<string | null>(null)
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all")
  const [focusedResultIndex, setFocusedResultIndex] = useState(0)
  const resultListRef = useRef<HTMLDivElement>(null)
  const pendingResultFocusRef = useRef<number | null>(null)

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // -- Partition results into conference-friendly outcome groups ------------

  const {
    completed,
    includedExisting,
    metadataUpdated,
    skippedExisting,
    submitFailed,
    processingFailed,
    metadataUpdateFailed,
    cancelled,
  } = useMemo(() => groupResultItems(results), [results])

  const successes = useMemo(
    () => [...completed, ...includedExisting, ...metadataUpdated],
    [completed, includedExisting, metadataUpdated]
  )

  const failures = useMemo(
    () => [...submitFailed, ...processingFailed, ...metadataUpdateFailed, ...cancelled],
    [submitFailed, processingFailed, metadataUpdateFailed, cancelled]
  )

  const filteredResults = useMemo(
    () =>
      resultFilter === "all"
        ? results
        : results.filter((item) => terminalOutcomeForResult(item) === resultFilter),
    [resultFilter, results]
  )
  const usesVirtualResults = results.length >= 100
  // TanStack Virtual exposes an imperative object that React Compiler skips.
  // eslint-disable-next-line react-hooks/incompatible-library
  const resultVirtualizer = useVirtualizer({
    count: usesVirtualResults ? filteredResults.length : 0,
    getScrollElement: () => resultListRef.current,
    estimateSize: () => 64,
    overscan: 6,
    getItemKey: (index) => filteredResults[index]?.id ?? index,
    measureElement: (element) => element?.getBoundingClientRect().height || 64,
  })
  const resultVirtualItems = resultVirtualizer.getVirtualItems()

  const focusResultIndex = useCallback(
    (requestedIndex: number) => {
      if (filteredResults.length === 0) return
      const nextIndex = Math.max(
        0,
        Math.min(requestedIndex, filteredResults.length - 1)
      )
      setFocusedResultIndex(nextIndex)
      const row = resultListRef.current?.querySelector<HTMLElement>(
        `[data-index="${nextIndex}"]`
      )
      if (row) {
        pendingResultFocusRef.current = null
        row.focus()
        return
      }
      pendingResultFocusRef.current = nextIndex
      resultVirtualizer.scrollToIndex(nextIndex)
    },
    [filteredResults.length, resultVirtualizer]
  )

  useEffect(() => {
    const pendingIndex = pendingResultFocusRef.current
    if (pendingIndex == null) return
    const row = resultListRef.current?.querySelector<HTMLElement>(
      `[data-index="${pendingIndex}"]`
    )
    if (!row) return
    pendingResultFocusRef.current = null
    row.focus()
  }, [resultVirtualItems])

  useEffect(() => {
    if (filteredResults.length === 0) {
      setFocusedResultIndex(0)
      return
    }
    if (focusedResultIndex >= filteredResults.length) {
      setFocusedResultIndex(filteredResults.length - 1)
    }
  }, [filteredResults.length, focusedResultIndex])

  const collectionId = tracking?.collectionId
  const hasDurableCollection =
    Boolean(collectionId) && tracking?.durableMode === "durable_collection"
  const readyCollectionItemCount = useMemo(
    () => [...successes, ...skippedExisting].filter(hasReadyMedia).length,
    [successes, skippedExisting]
  )
  const canAskCollection =
    hasDurableCollection &&
    readyCollectionItemCount > 0 &&
    Boolean(onSearchKnowledge) &&
    Boolean(capabilities?.hasKnowledgeQaMediaScope)
  const showGenericSearch =
    successes.length > 0 && Boolean(onSearchKnowledge) && !hasDurableCollection
  const hasWorkspaceOpenTarget =
    Boolean(onOpenWorkspace) &&
    successes.some((item) => item.persisted && shouldKeepOriginalFile(item.type))
  const showCollectionOpen =
    hasDurableCollection && Boolean(onOpenCollection) && Boolean(collectionId)
  const showNextSteps =
    showGenericSearch || hasWorkspaceOpenTarget || showCollectionOpen || canAskCollection

  // -- Classify each error --------------------------------------------------

  const errorCategories = useMemo(() => {
    const map = new Map<string, ErrorCategory>()
    for (const item of failures) {
      map.set(item.id, classifyError(item.error))
    }
    return map
  }, [failures])

  // -- Retryable error IDs --------------------------------------------------

  const conferenceRetryRequests = useMemo(
    () =>
      hasDurableCollection
        ? buildConferenceRetryRequestItems(
            failures.filter(
              (item) =>
                item.retryable ?? errorCategories.get(item.id)?.retryable
            )
          )
        : [],
    [errorCategories, failures, hasDurableCollection]
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
            .filter(
              (item) =>
                item.retryable ?? errorCategories.get(item.id)?.retryable
            )
            .map((e) => e.id),
    [conferenceRetryRequests, errorCategories, failures, hasDurableCollection]
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
      if (!onRetryItems) return undefined
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
    ]
  )

  const renderResultItem = useCallback(
    (item: WizardResultItem) => {
      const outcome = terminalOutcomeForResult(item)
      if (outcome === "skipped_existing") {
        return (
          <SkippedRow
            item={item}
            qi={qi}
            onOpenMedia={onOpenMedia}
            onDiscussInChat={onDiscussInChat}
          />
        )
      }
      if (
        outcome === "submit_failed" ||
        outcome === "processing_failed" ||
        outcome === "metadata_update_failed" ||
        outcome === "cancelled"
      ) {
        return (
          <ErrorRow
            item={item}
            category={errorCategories.get(item.id) ?? classifyError(item.error)}
            qi={qi}
            onRetry={getRetryHandlerForItem(item)}
          />
        )
      }
      return (
        <SuccessRow
          item={item}
          qi={qi}
          onOpenMedia={onOpenMedia}
          onDiscussInChat={onDiscussInChat}
        />
      )
    },
    [
      errorCategories,
      getRetryHandlerForItem,
      onDiscussInChat,
      onOpenMedia,
      qi,
    ]
  )

  const handleIngestMore = useCallback(() => {
    if (onIngestMore) {
      onIngestMore()
      return
    }
    reset()
  }, [onIngestMore, reset])

  const handleResultRowKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLElement>, index: number) => {
      let nextIndex: number | null = null
      switch (event.key) {
        case "ArrowDown":
          nextIndex = index + 1
          break
        case "ArrowUp":
          nextIndex = index - 1
          break
        case "Home":
          nextIndex = 0
          break
        case "End":
          nextIndex = filteredResults.length - 1
          break
      }
      if (nextIndex == null) return
      event.preventDefault()
      focusResultIndex(nextIndex)
    },
    [filteredResults.length, focusResultIndex]
  )

  const handleResultFilterChange = useCallback(
    (nextFilter: ResultFilter) => {
      setResultFilter(nextFilter)
      setFocusedResultIndex(0)
      pendingResultFocusRef.current = 0
      resultVirtualizer.scrollToIndex(0)
    },
    [resultVirtualizer]
  )

  // -- Elapsed time ---------------------------------------------------------

  const elapsedLabel = useMemo(
    () => formatElapsed(processingState.elapsed),
    [processingState.elapsed]
  )
  const showsOutcome = (outcome: TerminalOutcome): boolean =>
    resultFilter === "all" || resultFilter === outcome

  // -- Render ---------------------------------------------------------------

  return (
    <div className="flex h-full flex-col" data-testid="wizard-results-step">
      {/* Scrollable content area */}
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {results.length > 0 && (
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <label className="text-xs text-text-muted" htmlFor="quick-ingest-result-filter">
              {qi("wizard.results.filterLabel", "Show")}
            </label>
            <select
              id="quick-ingest-result-filter"
              aria-label={qi("wizard.results.filterAria", "Filter results by outcome")}
              className="rounded border border-border bg-surface px-2 py-1 text-xs text-text"
              value={resultFilter}
              onChange={(event) =>
                handleResultFilterChange(event.target.value as ResultFilter)
              }
            >
              <option value="all">{qi("wizard.results.filterAll", "All outcomes")}</option>
              <option value="completed">{qi("wizard.results.completed", "Completed")}</option>
              <option value="included_existing">{qi("wizard.results.includedExisting", "Included existing")}</option>
              <option value="metadata_updated">{qi("wizard.results.metadataUpdated", "Metadata updated")}</option>
              <option value="skipped_existing">{qi("wizard.results.skippedExisting", "Skipped existing")}</option>
              <option value="submit_failed">{qi("wizard.results.notSubmitted", "Not submitted")}</option>
              <option value="processing_failed">{qi("wizard.results.processingFailed", "Failed during processing")}</option>
              <option value="metadata_update_failed">{qi("wizard.results.metadataUpdateFailed", "Metadata update failed")}</option>
              <option value="cancelled">{qi("wizard.results.cancelled", "Cancelled")}</option>
            </select>
          </div>
        )}

        {/* Completed */}
        {completed.length > 0 && showsOutcome("completed") && (
          <section aria-label={qi("wizard.results.completedSection", "Completed items")}>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
              <Check className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
              {completed.some((item) => item.terminalOutcome)
                ? qi("wizard.results.completedHeading", "Completed ({{count}})", {
                    count: completed.length,
                  })
                : qi("wizard.results.succeededHeading", "Succeeded ({{count}})", {
                    count: completed.length,
                  })}
            </h3>
            {!usesVirtualResults && <div className="space-y-0.5">
              {completed.map((item) => (
                <SuccessRow
                  key={item.id}
                  item={item}
                  qi={qi}
                  onOpenMedia={onOpenMedia}
                  onDiscussInChat={onDiscussInChat}
                />
              ))}
            </div>}
          </section>
        )}

        {includedExisting.length > 0 && showsOutcome("included_existing") && (
          <section className="mt-4" aria-label={qi("wizard.results.includedExistingSection", "Included existing items")}>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
              <Check className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
              {qi("wizard.results.includedExistingHeading", "Included existing ({{count}})", { count: includedExisting.length })}
            </h3>
            {!usesVirtualResults && <div className="space-y-0.5">
              {includedExisting.map((item) => (
                <SuccessRow key={item.id} item={item} qi={qi} onOpenMedia={onOpenMedia} onDiscussInChat={onDiscussInChat} />
              ))}
            </div>}
          </section>
        )}

        {metadataUpdated.length > 0 && showsOutcome("metadata_updated") && (
          <section className="mt-4" aria-label={qi("wizard.results.metadataUpdatedSection", "Metadata updated items")}>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted">
              <Check className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
              {qi("wizard.results.metadataUpdatedHeading", "Metadata updated ({{count}})", { count: metadataUpdated.length })}
            </h3>
            {!usesVirtualResults && <div className="space-y-0.5">
              {metadataUpdated.map((item) => (
                <SuccessRow key={item.id} item={item} qi={qi} onOpenMedia={onOpenMedia} onDiscussInChat={onDiscussInChat} />
              ))}
            </div>}
          </section>
        )}

        {/* Next steps CTAs */}
        {showNextSteps && (
          <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3">
            <p className="mb-2 text-xs font-medium text-text-muted">
              {qi("wizard.results.nextSteps", "What's next?")}
            </p>
            <div className="flex flex-wrap gap-2">
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
                  onClick={onSearchKnowledge}
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
                  onClick={onSearchKnowledge}
                  className="flex items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 transition-colors"
                  aria-label={qi("wizard.results.searchKnowledgeAria", "Search your ingested content in Knowledge QA")}
                >
                  <Search className="h-3.5 w-3.5" aria-hidden="true" />
                  {qi("wizard.results.searchKnowledge", "Search in Knowledge")}
                </button>
              )}
              {hasWorkspaceOpenTarget && onOpenWorkspace && (
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
        {skippedExisting.length > 0 && showsOutcome("skipped_existing") && (
          <section
            aria-label={qi("wizard.results.skippedSection", "Skipped items")}
            className={successes.length > 0 ? "mt-4" : ""}
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" aria-hidden="true" />
              {qi("wizard.results.skippedExistingHeading", "Skipped existing ({{count}})", {
                count: skippedExisting.length,
              })}
            </h3>
            {!usesVirtualResults && <div className="space-y-1">
              {skippedExisting.map((item) => (
                <SkippedRow
                  key={item.id}
                  item={item}
                  qi={qi}
                  onOpenMedia={onOpenMedia}
                  onDiscussInChat={onDiscussInChat}
                />
              ))}
            </div>}
          </section>
        )}

        {/* Failure export/retry actions */}
        {failures.length > 0 && (
          <div
            className={
              successes.length > 0 || skippedExisting.length > 0
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
          </div>
        )}

        {/* Submission failures */}
        {submitFailed.length > 0 && showsOutcome("submit_failed") && (
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
            {!usesVirtualResults && <div className="space-y-2">
              {submitFailed.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                />
              ))}
            </div>}
          </section>
        )}

        {/* Processing failures */}
        {processingFailed.length > 0 && showsOutcome("processing_failed") && (
          <section
            aria-label={qi("wizard.results.errorsSection", "Error items")}
            className="mt-4"
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-danger">
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              {qi("wizard.results.failedProcessingHeading", "Failed during processing ({{count}})", {
                count: processingFailed.length,
              })}
            </h3>
            {!usesVirtualResults && <div className="space-y-2">
              {processingFailed.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                />
              ))}
            </div>}
          </section>
        )}

        {/* Metadata update failures */}
        {metadataUpdateFailed.length > 0 && showsOutcome("metadata_update_failed") && (
          <section
            aria-label={qi("wizard.results.metadataUpdateFailedSection", "Metadata update failed items")}
            className="mt-4"
          >
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-danger">
              <X className="h-3.5 w-3.5" aria-hidden="true" />
              {qi("wizard.results.metadataUpdateFailedHeading", "Metadata update failed ({{count}})", {
                count: metadataUpdateFailed.length,
              })}
            </h3>
            {!usesVirtualResults && <div className="space-y-2">
              {metadataUpdateFailed.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                />
              ))}
            </div>}
          </section>
        )}

        {/* Cancelled */}
        {cancelled.length > 0 && showsOutcome("cancelled") && (
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
            {!usesVirtualResults && <div className="space-y-2">
              {cancelled.map((item) => (
                <ErrorRow
                  key={item.id}
                  item={item}
                  category={errorCategories.get(item.id) ?? classifyError(item.error)}
                  qi={qi}
                  onRetry={getRetryHandlerForItem(item)}
                />
              ))}
            </div>}
          </section>
        )}

        {usesVirtualResults && (
          <div
            ref={resultListRef}
            className="mt-4 max-h-[50vh] overflow-y-auto"
            role="list"
            aria-label={qi("wizard.results.terminalListAria", "Terminal results")}
          >
            <div className="relative w-full" style={{ height: resultVirtualizer.getTotalSize() }}>
              {resultVirtualItems.map((virtualRow) => {
                const item = filteredResults[virtualRow.index]
                if (!item) return null
                const outcome = terminalOutcomeForResult(item)
                return (
                  <div
                    key={virtualRow.key}
                    ref={resultVirtualizer.measureElement}
                    role="listitem"
                    tabIndex={virtualRow.index === focusedResultIndex ? 0 : -1}
                    aria-setsize={filteredResults.length}
                    aria-posinset={virtualRow.index + 1}
                    data-terminal-outcome={outcome}
                    data-index={virtualRow.index}
                    onFocus={() => setFocusedResultIndex(virtualRow.index)}
                    onKeyDown={(event) =>
                      handleResultRowKeyDown(event, virtualRow.index)
                    }
                    className="absolute left-0 top-0 w-full pb-1"
                    style={{ transform: `translateY(${virtualRow.start}px)` }}
                  >
                    <span className="mb-1 block px-3 text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                      {terminalOutcomeLabel(outcome, qi)}
                    </span>
                    {renderResultItem(item)}
                  </div>
                )
              })}
            </div>
          </div>
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
            {skippedExisting.length > 0 ||
            submitFailed.length > 0 ||
            cancelled.length > 0
              ? qi(
                  "wizard.results.summaryWithOutcomeGroups",
                  "Total: {{success}} succeeded, {{skipped}} skipped, {{notSubmitted}} not submitted, {{failed}} failed, {{cancelled}} cancelled",
                  {
                    success: successes.length,
                    skipped: skippedExisting.length,
                    notSubmitted: submitFailed.length,
                    failed: processingFailed.length + metadataUpdateFailed.length,
                    cancelled: cancelled.length,
                  }
                )
              : qi(
                  "wizard.results.summary",
                  "Total: {{success}} succeeded, {{failed}} failed",
                  {
                    success: successes.length,
                    failed: processingFailed.length + metadataUpdateFailed.length,
                  }
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
