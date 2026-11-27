import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
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
import { canPasteQueueItem, getQueueItemExclusionReason, getEligibleQueueItems } from "./queue-items"
import type { ItemProgress, WizardQueueItem, WizardResultItem } from "./types"
import { getExtractedSourceText } from "./processing-results"
import { ExtractedTextViewer } from "./ExtractedTextViewer"
import { shouldKeepOriginalFile } from "@/services/tldw/media-routing"
import {
  buildConferenceFailedResultExportText,
  buildConferenceRetryRequestItems,
  UNDISPATCHED_PLAN_SOURCE,
  type ConferenceRetryRequestItem,
} from "@/services/tldw/conference-collections"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { useIngestWizard } from "./IngestWizardContext"
import { classifyError, extractionFailureSources } from "./ErrorClassification"
import type { ErrorCategory } from "./ErrorClassification"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"
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
  oneScreen?: boolean
  reviewDrafts?: { count: number; onOpen: () => void }
  interruption?: {
    sources: Array<{ id: string; name: string; jobIds: number[]; lastProgress?: ItemProgress }>
    onCheckMedia: () => void
    onCheckStatus?: () => void
    checkError?: string | null
    onResumeSetup?: () => void
  }
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
  onCorrectItems?: (ids: string[], repairSource?: boolean) => void
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
  oneScreen?: boolean
  reviewFirst?: boolean
  extractedText?: string
  onViewText?: (item: WizardResultItem, trigger: HTMLButtonElement) => void
  item: WizardResultItem
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
}

const SuccessRow: React.FC<SuccessRowProps> = React.memo(
  ({ item, qi, onOpenMedia, onDiscussInChat, oneScreen, reviewFirst, extractedText, onViewText }) => {
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
            : reviewFirst ? qi("web.extractedForReview", "Extracted for review")
              : qi("wizard.results.extractedState", "Extracted content; not saved")}
            {item.data && typeof item.data === "object" && "analysis" in item.data && typeof item.data.analysis === "string" && item.data.analysis.trim()
              ? ` · ${qi("wizard.results.analysisAvailable", "Analysis available")}` : ""}</p>
          {!oneScreen && getSavedMediaIds([item]).length > 0 && <p className="text-xs text-text-muted">{qi("wizard.results.knowledgeUnconfirmed", "Knowledge readiness unconfirmed")}</p>}
          {item.warning && <p className="whitespace-pre-line break-words text-xs text-warn">{item.warning}</p>}
          {extractedText === "" && <p className="mt-1 text-xs text-text-muted">{item.data == null
            ? qi("web.extractionUnavailable", "Extracted text is unavailable in this receipt.")
            : qi("web.emptyExtraction", "No readable text was returned for this source.")}</p>}
        </div>
        {duration && (
          <span className="flex-shrink-0 text-xs tabular-nums text-text-muted">
            {duration}
          </span>
        )}
        <div className="flex flex-shrink-0 items-center gap-1">
          {extractedText && onViewText && (
            <button type="button" onClick={event => onViewText(item, event.currentTarget)}
              className="min-h-11 rounded px-2 text-sm font-medium text-primary hover:bg-primary/10"
              aria-label={qi("web.viewTextFor", "View text for {{name}}", { name: label })}>
              {qi("web.viewText", "View text")}
            </button>
          )}
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
  localDuplicateMessage: string
  item: WizardResultItem
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onOpenMedia?: (item: WizardResultItem) => void
  onDiscussInChat?: (item: WizardResultItem) => void
}

const SkippedRow: React.FC<SkippedRowProps> = React.memo(
  ({ item, qi, onOpenMedia, onDiscussInChat, localDuplicateMessage }) => {
    const label = item.title || item.fileName || item.url || item.id
    const showOpenMedia = Boolean(onOpenMedia) && canOpenMedia(item)
    const skippedReason = resolveSkippedResultReason(item)
    const skippedMessage =
      skippedReason === "local-queue-duplicate"
        ? localDuplicateMessage
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
          {item.warning && (
            <p className="mt-0.5 whitespace-pre-line break-words text-xs text-text-subtle">{item.warning}</p>
          )}
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
  onCorrect?: (repairSource?: boolean) => void
  onReattach?: (file: File) => void
  source?: WizardQueueItem
  oneScreen: boolean
}

const MAX_LISTED_FAILED_SOURCES = 5

function sourceRepairLabel(source: WizardQueueItem, qi: ErrorRowProps["qi"]): string {
  if (source.url) return qi("source.edit", "Edit link")
  if (canPasteQueueItem(source)) return qi("source.pasteAgain", "Paste again")
  return qi("source.chooseFile", "Choose file")
}

const ErrorRow: React.FC<ErrorRowProps> = React.memo(
  ({ item, category, qi, onRetry, onRemove, onCorrect, onReattach, source, oneScreen }) => {
    const label = item.title || item.fileName || item.url || item.id
    const missingFile = source && !source.url && !source.file
    const settingsFailure = (
      !category.retryable && /\b(?:api_name|api_provider|analysis provider|transcription model)\b/i.test(item.error ?? "")
    )
    const failedSources = extractionFailureSources(item.data)
    const rejectedSource = !settingsFailure && (category.classification === "validation" || (!category.retryable && failedSources.length > 0) || /\b(?:404|410|415|422)\b/.test(item.error ?? ""))
    const repairLabel = oneScreen && source && (rejectedSource || missingFile)
      ? sourceRepairLabel(source, qi) : null
    const showFilePicker = missingFile && onReattach && (!oneScreen || ((category.retryable || settingsFailure) && source.fileOrigin !== "paste"))
    const reattachError = source?.validation.errors?.join("; ")
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
              {oneScreen && rejectedSource && source?.url && !failedSources.length
                ? qi("source.rejectedLink", "The server could not process this link. Check the address and try again.")
                : <>{category.userMessage} {category.suggestion}</>}
            </p>
            {item.error && (
              <details className="mt-1 text-xs text-text-subtle">
                <summary className="cursor-pointer">{qi("wizard.results.errorDetails", "Error details")}</summary>
                <p className="mt-1 break-words">{sanitizeServerErrorMessage(item.error, category.userMessage)}</p>
              </details>
            )}
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
            {showFilePicker && <label className="text-xs text-text">
              {qi("wizard.results.reattach", "Reattach {{name}}", { name: label })}
              <input type="file" aria-label={qi("wizard.results.reattach", "Reattach {{name}}", { name: label })}
                className="mt-1 block max-w-full text-xs" onChange={event => { const file = event.target.files?.[0]; if (file) onReattach(file); event.target.value = "" }} />
              {reattachError && <span role="alert" className="block text-danger">{reattachError}</span>}
            </label>}
            {onCorrect && repairLabel && (!showFilePicker || (source && canPasteQueueItem(source))) && (
              <button type="button" onClick={() => onCorrect(true)} className="rounded px-2 py-1 text-sm text-primary hover:bg-primary/10">
                {repairLabel}
              </button>
            )}
            {onCorrect && !onRetry && !showFilePicker && !repairLabel && (!oneScreen || (source && settingsFailure)) && <button type="button" onClick={() => onCorrect()}
              className="rounded px-2 py-1 text-xs text-primary hover:bg-primary/10"
              aria-label={oneScreen ? qi("source.editSettingsFor", "Edit settings for {{name}}", { name: label }) : qi("wizard.results.correctItem", "Correct settings for {{name}}", { name: label })}>
              {oneScreen ? qi("source.editSettings", "Edit settings") : qi("wizard.results.correct", "Correct settings")}
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
  oneScreen = false,
  reviewDrafts,
  interruption,
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
  const excludedItems = queueItems.filter(item =>
    getQueueItemExclusionReason(item, queueItems) &&
    !results.some(result => result.id === item.id) &&
    !interruption?.sources.some(source => source.id === item.id)
  )
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
  const extractionTexts = useMemo(() => new Map(oneScreen
    ? savedItems.filter(item => !getSavedMediaIds([item]).length)
      .map(item => [item.id, getExtractedSourceText(item.data)] as const)
    : []), [oneScreen, savedItems])
  const [viewedItem, setViewedItem] = useState<WizardResultItem | null>(null)
  const viewTrigger = useRef<HTMLButtonElement | null>(null)
  const openExtraction = (item: WizardResultItem, trigger: HTMLButtonElement) => {
    viewTrigger.current = trigger
    setViewedItem(item)
  }
  useEffect(() => {
    if (!viewedItem && viewTrigger.current) {
      viewTrigger.current.focus()
      viewTrigger.current = null
    }
  }, [viewedItem])

  const failures = useMemo(
    () => [...submitFailed, ...failedProcessing, ...cancelled],
    [submitFailed, failedProcessing, cancelled]
  )

  const collectionId = tracking?.collectionId
  const hasDurableCollection =
    Boolean(collectionId) && tracking?.durableMode === "durable_collection"
  // A skipped library duplicate can open existing media but was not saved by this run.
  const newlySavedIds = getSavedMediaIds(savedItems).filter(id => savedIds.includes(id))
  const savedMediaIds = newlySavedIds
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
      if ((item.data as { code?: string } | undefined)?.code === UNDISPATCHED_PLAN_SOURCE) {
        map.set(item.id, { ...category,
          badgeLabel: qi("web.planReadyBadge", "Ready to submit"),
          userMessage: qi("web.planReadyMessage", "This source can resume setup."),
          suggestion: qi("web.planReadySuggestion", "Retry to submit it to the original collection."),
        })
        continue
      }
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

  const renderErrorRow = (item: WizardResultItem) => (
    <ErrorRow
      key={item.id} item={item} oneScreen={oneScreen}
      source={queueItems.find(source => source.id === item.id)}
      category={errorCategories.get(item.id) ?? classifyError(item.error, item.data)}
      qi={qi}
      onRetry={getRetryHandlerForItem(item)}
      onCorrect={onCorrectItems ? repair => repair ? onCorrectItems([item.id], true) : onCorrectItems([item.id]) : undefined}
      onReattach={onReattach ? file => onReattach(item.id, file) : undefined}
    />
  )
  const handleIngestMore = useCallback(() => {
    if (onIngestMore) {
      onIngestMore()
      return
    }
    reset()
  }, [onIngestMore, reset])

  const reviewSaved = (ids: Array<string | number>) => {
    void Promise.resolve(onReviewSavedItems?.(ids)).catch(() =>
      setExportNotice(qi("wizard.results.savedReviewFailed", "Unable to open saved items. Try again."))
    )
  }

  // -- Elapsed time ---------------------------------------------------------

  const elapsedLabel = useMemo(
    () => formatElapsed(processingState.elapsed),
    [processingState.elapsed]
  )

  // -- Render ---------------------------------------------------------------

  const singleSaved = newlySavedIds.length === 1
    ? savedItems.find(item => getSavedMediaIds([item]).includes(newlySavedIds[0]))
    : undefined
  const extractedCount = savedItems.filter(item => !getSavedMediaIds([item]).length).length
  const failedCount = submitFailed.length + failedProcessing.length
  const skippedCount = skippedExisting.length + excludedItems.length
  const confirmedDrafts = reviewDrafts && reviewDrafts.count > 0 ? reviewDrafts : undefined
  const localDuplicateMessage = oneScreen
    ? qi("source.duplicate", "Duplicate · skipped")
    : qi("wizard.results.skippedAlreadyQueued", LOCAL_QUEUE_DUPLICATE_SKIP_MESSAGE)
  const reviewFirst = oneScreen && state.presetConfig.reviewBeforeStorage
  const outcomeLabels = oneScreen ? [
    confirmedDrafts && (confirmedDrafts.count === 1 ? qi("web.resultDraft", "1 draft ready for review")
      : qi("web.resultDrafts", "{{count}} drafts ready for review", { count: confirmedDrafts.count })),
    newlySavedIds.length > 0 && qi("web.resultSaved", "{{count}} saved to Media", { count: newlySavedIds.length }),
    extractedCount > 0 && !confirmedDrafts && (reviewFirst
      ? qi("web.resultExtractedForReview", "{{count}} extracted for review", { count: extractedCount })
      : qi("web.resultExtracted", "{{count}} extracted", { count: extractedCount })),
    failedCount > 0 && qi("web.resultFailed", "{{count}} failed", { count: failedCount }),
    skippedCount > 0 && qi("web.resultSkipped", "{{count}} skipped", { count: skippedCount }),
    cancelled.length > 0 && qi("web.resultCancelled", "{{count}} cancelled", { count: cancelled.length }),
  ].filter(Boolean) : []

  let primaryAction: { kind: "retry" | "drafts" | "open" | "review" | "extract"; label: string; run: React.MouseEventHandler<HTMLButtonElement> } | undefined
  if (oneScreen) {
    if (retryableIds.length && onRetryItems) {
      primaryAction = {
        kind: "retry",
        label: retryableIds.length === 1 ? qi("web.retrySource", "Retry failed source")
          : qi("web.retrySources", "Retry {{count}} failed sources", { count: retryableIds.length }),
        run: handleRetryAll,
      }
    } else if (confirmedDrafts) {
      primaryAction = { kind: "drafts", label: qi("web.openDrafts", "Open drafts"), run: confirmedDrafts.onOpen }
    } else if (singleSaved && onOpenMedia) {
      primaryAction = { kind: "open", label: qi("wizard.results.open", "Open in Media"), run: () => onOpenMedia(singleSaved) }
    } else if (newlySavedIds.length > 1 && onReviewSavedItems) {
      primaryAction = {
        kind: "review",
        label: qi("web.reviewSources", "Review {{count}} saved sources", { count: newlySavedIds.length }),
        run: () => reviewSaved(newlySavedIds),
      }
    } else if (results.length === 1 && extractionTexts.get(results[0].id)) {
      primaryAction = {
        kind: "extract",
        label: qi("web.viewExtractedText", "View extracted text"),
        run: event => openExtraction(results[0], event.currentTarget),
      }
    }
  }
  const ActionsPanel = oneScreen ? "details" : "div"
  const ActionsTitle = oneScreen ? "summary" : "p"
  const primaryStyle = "rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
  const secondaryStyle = "rounded-md border border-border bg-surface px-4 py-2 text-sm font-medium text-text hover:bg-surface2"

  const nextSteps = (showNextSteps || oneScreen) && (
    <ActionsPanel className={oneScreen ? "my-3 border-y border-border py-2" : "mt-4 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3"}>
      <ActionsTitle className={oneScreen ? "cursor-pointer py-1 text-sm font-medium text-text" : "mb-2 text-xs font-medium text-text-muted"}>
        {oneScreen ? qi("web.moreActions", "More actions") : qi("wizard.results.nextSteps", "What's next?")}
      </ActionsTitle>
      <div className={oneScreen ? "mt-2 flex flex-wrap gap-2" : "flex flex-wrap gap-2"}>
        {oneScreen && confirmedDrafts && primaryAction?.kind !== "drafts" && (
          <button type="button" onClick={confirmedDrafts.onOpen} className={secondaryStyle}>
            {qi("web.openDrafts", "Open drafts")}
          </button>
        )}
        {savedIds.length > 0 && onReviewSavedItems && (primaryAction?.kind !== "review" || newlySavedIds.length !== savedIds.length) && <button type="button" onClick={() => reviewSaved(savedIds)} className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text hover:bg-surface2">
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
        {oneScreen && (
          <button type="button" onClick={handleIngestMore} className={secondaryStyle}>
            {qi("web.ingestMore", "Add more sources")}
          </button>
        )}
        {oneScreen && failures.length > 0 && (
          <button type="button" onClick={handleExportFailedItems} className={secondaryStyle}>
            {qi("wizard.results.exportFailedListAria", "Export failed items list")}
          </button>
        )}
      </div>
      {oneScreen && (showGenericSearch || canAskCollection) && (
        <p className="mt-2 text-xs text-text-muted">{qi("web.searchReadiness", "Availability in Knowledge search depends on source readiness.")}</p>
      )}
    </ActionsPanel>
  )

  return (
    <>
    {viewedItem && <ExtractedTextViewer item={viewedItem}
      text={extractionTexts.get(viewedItem.id) ?? ""} qi={qi} onBack={() => setViewedItem(null)} />}
    <div hidden={Boolean(viewedItem)} className={viewedItem ? "hidden" : "flex h-full min-h-0 flex-col"} data-testid="wizard-results-step">
      {/* Scrollable content area */}
      <div className={oneScreen ? "qi-web-scroll min-h-0 flex-1 overflow-y-auto" : "min-h-0 flex-1 overflow-y-auto px-4 py-3"}>
        {oneScreen ? <div role="status" className="mb-4">
          <h2 className="text-xl font-semibold text-text">{outcomeLabels[0] || (interruption ? qi("web.statusUnavailable", "Status unavailable") : qi("web.resultsTitle", "Import results"))}</h2>
          {outcomeLabels.length > 1 && <p className="mt-1 text-sm text-text-muted">{outcomeLabels.slice(1).join(" · ")}</p>}
        </div> : <p role="status" className="mb-3 text-sm text-text">
          {qi("wizard.results.batchAccounting", "{{added}} added · {{excluded}} excluded/skipped · {{succeeded}} succeeded ({{saved}} saved) · {{failed}} failed · {{cancelled}} cancelled", {
            added: new Set([...queueItems.map(item => item.id), ...results.map(item => item.id)]).size,
            excluded: excludedItems.length + skippedExisting.length,
            succeeded: savedItems.length, saved: savedIds.length,
            failed: submitFailed.length + failedProcessing.length, cancelled: cancelled.length,
          })}
        </p>}
        {interruption && <section aria-label={qi("web.statusUnavailable", "Status unavailable")} className="mb-4">
          <p role="alert" className="text-sm text-text">
            {qi("web.unconfirmedHelp", "The server could not confirm the outcome. Check before importing these sources again.")}
          </p>
          <ul className="my-3 space-y-2">
            {interruption.sources.map(source => <li key={source.id} className="break-words text-sm text-text">
              {source.name}
              {source.lastProgress && source.lastProgress.status !== "unknown" && <span className="block text-xs text-text-muted">
                {qi("web.lastReported", "Last reported: {{stage}}", {
                  stage: source.lastProgress.currentStage || qi("processing.status." + source.lastProgress.status, source.lastProgress.status),
                })}
                {source.lastProgress.progressPercent > 0 && ` · ${source.lastProgress.progressPercent}%`}
              </span>}
              {source.jobIds.length > 0 && <span className="block text-xs text-text-muted">
                {qi("web.trackedJobs", "Job IDs: {{ids}}", { ids: source.jobIds.join(", ") })}
              </span>}
            </li>)}
          </ul>
          <div className="flex flex-wrap gap-2">
            {interruption.onCheckStatus && <button type="button" className={primaryStyle} onClick={interruption.onCheckStatus}>{qi("web.checkStatus", "Check status")}</button>}
            {interruption.onResumeSetup && <button type="button" className={secondaryStyle} onClick={interruption.onResumeSetup}>{qi("web.resumeSetup", "Resume original collection setup")}</button>}
            <button type="button" className={secondaryStyle} onClick={interruption.onCheckMedia}>{qi("web.checkMedia", "Check Media")}</button>
          </div>
          {interruption.checkError && <p role="alert" className="mt-2 text-sm text-text">{interruption.checkError}</p>}
        </section>}
        {excludedItems.length > 0 && <section aria-label={qi("wizard.results.excludedSection", "Excluded inputs")} className="mb-4 space-y-2">
          {excludedItems.map(item => <div key={item.id} className="text-sm text-text">
            <p className="break-words">{item.fileName || item.url || item.id}</p>
            <p className="text-xs text-text-muted">{getQueueItemExclusionReason(item, queueItems) === "invalid"
              ? item.validation.errors?.join("; ") || qi("wizard.results.invalidExcluded", "Invalid input. Correct it before processing.")
              : getQueueItemExclusionReason(item, queueItems) === "unselected"
                ? qi("wizard.results.unselectedExcluded", "Not selected for this run.")
                : localDuplicateMessage}</p>
            {oneScreen && onCorrectItems && !item.validation.valid && item.conferenceOverride?.selected !== false && (
              <button type="button" onClick={() => onCorrectItems([item.id], true)}
                className="rounded px-2 py-1 text-sm text-primary hover:bg-primary/10">
                {sourceRepairLabel(item, qi)}
              </button>
            )}
          </div>)}
        </section>}
        {/* Successes */}
        {successes.length > 0 && (
          <section aria-label={qi("wizard.results.completedSection", "Completed items")}>
            <h3 className={oneScreen ? "sr-only" : "mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-muted"}>
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
                  oneScreen={oneScreen}
                  reviewFirst={reviewFirst}
                  extractedText={extractionTexts.get(item.id)}
                  onViewText={primaryAction?.kind === "extract" ? undefined : openExtraction}
                  onOpenMedia={primaryAction?.kind === "open" && singleSaved === item ? undefined : onOpenMedia}
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
                <SuccessRow key={item.id} item={item} qi={qi} oneScreen={oneScreen} reviewFirst={reviewFirst}
                  extractedText={extractionTexts.get(item.id)}
                  onViewText={primaryAction?.kind === "extract" ? undefined : openExtraction}
                  onOpenMedia={primaryAction?.kind === "open" && singleSaved === item ? undefined : onOpenMedia}
                  onDiscussInChat={onDiscussInChat} />
              ))}
            </div>
          </section>
        )}

        {!oneScreen && nextSteps}

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
                  localDuplicateMessage={localDuplicateMessage}
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
        {!oneScreen && failures.length > 0 && (
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
              {submitFailed.map(renderErrorRow)}
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
              {failedProcessing.map(renderErrorRow)}
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
              {cancelled.map(renderErrorRow)}
            </div>
          </section>
        )}

        {/* Empty state */}
        {oneScreen && nextSteps}
        {oneScreen && exportNotice && <p role="status" className="text-sm text-text-muted">{exportNotice}</p>}
        {results.length === 0 && !interruption && (
          <div className="flex flex-col items-center justify-center py-12 text-center text-text-muted">
            <AlertTriangle className="mb-2 h-8 w-8 opacity-40" aria-hidden="true" />
            <p className="text-sm">
              {qi("wizard.results.noResults", "No results to display.")}
            </p>
          </div>
        )}
      </div>

      {/* Footer summary + actions */}
      {oneScreen ? (
        <div className="qi-web-footer flex flex-wrap items-center justify-end gap-2" role="group" aria-label={qi("web.resultActions", "Result actions")}>
          <button type="button" onClick={onClose} className={primaryAction ? secondaryStyle : primaryStyle}>
            {qi("wizard.results.done", "Done")}
          </button>
          {primaryAction && <button type="button" onClick={primaryAction.run} className={primaryStyle}>
            {primaryAction.label}
          </button>}
        </div>
      ) : results.length > 0 && (
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
    </>
  )
}

WizardResultsStep.displayName = "WizardResultsStep"

export default WizardResultsStep
