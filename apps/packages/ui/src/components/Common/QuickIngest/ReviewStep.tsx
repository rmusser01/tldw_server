import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useVirtualizer } from "@tanstack/react-virtual"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import {
  AlertTriangle,
  ArrowLeft,
  Play,
  FileText,
  Music,
  Film,
  Globe,
  Image,
  BookOpen,
  File,
} from "lucide-react"
import type { DetectedMediaType, IngestPreset, PresetConfig, WizardQueueItem } from "./types"
import {
  buildPlaylistIngestRunRequest,
  MAX_PLAYLIST_RUN_INPUTS,
  useIngestWizard,
} from "./IngestWizardContext"
import { estimateTotalSeconds, formatEstimate } from "./timeEstimation"
import { ItemMetadataTable } from "./ItemMetadataTable"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const LARGE_FILE_THRESHOLD = 50 * 1024 * 1024 // 50 MB
const LONG_TIME_THRESHOLD = 15 * 60 // 15 minutes in seconds
const LARGE_BATCH_THRESHOLD = 5

type ReviewFilter = "selected" | "duplicates" | "policy"

/**
 * Return a human-readable file size string (e.g., "42 MB", "1.2 GB").
 */
const formatFileSize = (bytes: number): string => {
  if (bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / Math.pow(1024, exp)
  // Show one decimal place for GB+, no decimals for smaller units
  const formatted = exp >= 3 ? value.toFixed(1) : Math.round(value).toString()
  return `${formatted} ${units[exp]}`
}

/**
 * Derive a human-readable description of operations that will be performed
 * on an item based on its detected type and the active preset configuration.
 */
const getOperationDescription = (
  type: DetectedMediaType,
  _preset: IngestPreset,
  config: PresetConfig
): string => {
  const parts: string[] = []

  // Type-specific operations
  if (type === "audio" || type === "video") parts.push("Transcribe")
  if (type === "document" || type === "pdf" || type === "ebook") {
    if (config.typeDefaults?.document?.ocr) parts.push("OCR")
    parts.push("Extract")
  }
  if (type === "web") parts.push("Scrape")
  if (type === "image") parts.push("Extract")

  // Common operations
  if (config.common.perform_analysis) parts.push("Analyze")
  if (config.common.perform_chunking) parts.push("Chunk")

  return parts.join(" + ") || "Process"
}

/**
 * Map a detected media type to the appropriate lucide icon component.
 */
const TYPE_ICONS: Record<DetectedMediaType, React.ElementType> = {
  audio: Music,
  video: Film,
  document: FileText,
  pdf: FileText,
  ebook: BookOpen,
  image: Image,
  web: Globe,
  unknown: File,
}

const getQueueItemOccurrenceId = (item: WizardQueueItem): string =>
  item.sourceRef?.occurrenceId || item.id

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type ReviewStepProps = {
  isOnlineForIngest?: boolean
  isCheckingConnection?: boolean
  connectionRecoveryMessage?: string
  onRetryConnection?: () => void
}

export const ReviewStep: React.FC<ReviewStepProps> = ({
  isOnlineForIngest = true,
  isCheckingConnection = false,
  connectionRecoveryMessage,
  onRetryConnection,
}) => {
  const { t } = useTranslation(["option"])
  const { state, goBack, startProcessing } = useIngestWizard()

  const [filter, setFilter] = useState<ReviewFilter>("selected")
  const reviewListRef = useRef<HTMLDivElement | null>(null)
  const reviewRowRefs = useRef(new Map<string, HTMLDivElement>())
  const reviewListOwnsFocusRef = useRef(false)
  const activeReviewRowRef = useRef<{ id: string; index: number } | null>(null)
  const [activeReviewId, setActiveReviewId] = useState<string | null>(null)

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  const { queueItems, selectedPreset, presetConfig, conferenceBatchMetadata } = state
  const selectedQueueItems = useMemo(
    () =>
      queueItems.filter(
        (item) => item.conferenceOverride?.selected !== false && item.playlistReview?.selected !== false
      ),
    [queueItems]
  )
  const filteredReviewItems = useMemo(
    () =>
      queueItems.filter((item) => {
        const selected =
          item.conferenceOverride?.selected !== false && item.playlistReview?.selected !== false
        const duplicate =
          item.playlist?.duplicateStatus === "duplicate_existing" ||
          item.playlist?.duplicateStatus === "duplicate_in_batch" ||
          item.playlistReview?.duplicateEvidence?.kind === "library" ||
          item.playlistReview?.duplicateEvidence?.kind === "in_run"
        if (filter === "duplicates") return duplicate
        if (filter === "policy") return Boolean(item.playlistReview?.duplicatePolicy)
        return selected
      }),
    [filter, queueItems]
  )
  const visibleReviewItemIds = useMemo(
    () => new Set(filteredReviewItems.map((item) => item.id)),
    [filteredReviewItems]
  )
  // TanStack Virtual exposes an imperative object that React Compiler skips.
  // eslint-disable-next-line react-hooks/incompatible-library
  const reviewVirtualizer = useVirtualizer({
    count: filteredReviewItems.length,
    getScrollElement: () => reviewListRef.current,
    estimateSize: () => 62,
    overscan: 6,
    getItemKey: (index) => filteredReviewItems[index]?.id ?? index,
    measureElement: (element) => element?.getBoundingClientRect().height || 62,
  })
  const reviewVirtualItems = reviewVirtualizer.getVirtualItems()
  const restoreReviewRowFocus = useCallback((id: string) => {
    const attempt = (remaining: number) => {
      if (!reviewListOwnsFocusRef.current) return
      const row = reviewRowRefs.current.get(id)
      if (row) {
        row.focus()
        return
      }
      if (remaining > 0) window.requestAnimationFrame(() => attempt(remaining - 1))
    }
    window.requestAnimationFrame(() => attempt(2))
  }, [])

  useEffect(() => {
    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target
      if (target instanceof Node && reviewListRef.current?.contains(target)) return
      reviewListOwnsFocusRef.current = false
    }
    document.addEventListener("focusin", handleFocusIn)
    return () => document.removeEventListener("focusin", handleFocusIn)
  }, [])

  useEffect(() => {
    if (filteredReviewItems.length === 0) {
      activeReviewRowRef.current = null
      setActiveReviewId(null)
      return
    }
    const active = activeReviewRowRef.current
    if (!active) {
      const id = getQueueItemOccurrenceId(filteredReviewItems[0])
      activeReviewRowRef.current = { id, index: 0 }
      setActiveReviewId(id)
      return
    }
    const currentIndex = filteredReviewItems.findIndex(
      (item) => getQueueItemOccurrenceId(item) === active.id
    )
    if (currentIndex >= 0) {
      active.index = currentIndex
      if (
        reviewListOwnsFocusRef.current &&
        !reviewRowRefs.current.has(active.id) &&
        reviewVirtualItems.length > 0
      ) {
        const nearest = reviewVirtualItems.reduce((best, row) =>
          Math.abs(row.index - currentIndex) < Math.abs(best.index - currentIndex) ? row : best
        )
        const nearestItem = filteredReviewItems[nearest.index]
        if (nearestItem) {
          const id = getQueueItemOccurrenceId(nearestItem)
          activeReviewRowRef.current = { id, index: nearest.index }
          setActiveReviewId(id)
          restoreReviewRowFocus(id)
        }
      }
      return
    }
    const index = Math.min(active.index, filteredReviewItems.length - 1)
    const id = getQueueItemOccurrenceId(filteredReviewItems[index])
    activeReviewRowRef.current = { id, index }
    setActiveReviewId(id)
    if (reviewListOwnsFocusRef.current) {
      reviewVirtualizer.scrollToIndex(index, { align: "auto" })
      restoreReviewRowFocus(id)
    }
  }, [filteredReviewItems, restoreReviewRowFocus, reviewVirtualItems, reviewVirtualizer])

  const handleReviewRowKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>, index: number) => {
      if (event.target !== event.currentTarget) return
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
      event.preventDefault()
      const targetIndex = Math.max(
        0,
        Math.min(
          filteredReviewItems.length - 1,
          index + (event.key === "ArrowDown" ? 1 : -1)
        )
      )
      const target = filteredReviewItems[targetIndex]
      if (!target) return
      const id = getQueueItemOccurrenceId(target)
      activeReviewRowRef.current = { id, index: targetIndex }
      setActiveReviewId(id)
      reviewVirtualizer.scrollToIndex(targetIndex, { align: "auto" })
      restoreReviewRowFocus(id)
    },
    [filteredReviewItems, restoreReviewRowFocus, reviewVirtualizer]
  )

  // Compute total estimated time
  const totalEstimatedSeconds = useMemo(
    () => estimateTotalSeconds(selectedQueueItems, selectedPreset),
    [selectedQueueItems, selectedPreset]
  )

  const estimatedTimeLabel = useMemo(
    () => formatEstimate(totalEstimatedSeconds),
    [totalEstimatedSeconds]
  )

  // Preset display name
  const presetLabel = useMemo(
    () => selectedPreset.charAt(0).toUpperCase() + selectedPreset.slice(1),
    [selectedPreset]
  )

  // Storage mode
  const storageMode = presetConfig.storeRemote ? "Server" : "Local"
  const validItemCount = useMemo(
    () => selectedQueueItems.filter((item) => item.validation.valid).length,
    [selectedQueueItems]
  )
  const runRequestBuild = useMemo(() => buildPlaylistIngestRunRequest(queueItems), [queueItems])
  const currentProcessingBlock = state.processingBlock ?? runRequestBuild.block
  const materializationExpired = currentProcessingBlock?.code === "materialization_expired"
  const reviewRequired = currentProcessingBlock?.code === "review_required"
  const exceedsRunInputLimit =
    currentProcessingBlock?.code === "invalid_run_request" &&
    validItemCount > MAX_PLAYLIST_RUN_INPUTS
  const canStartProcessing =
    validItemCount > 0 &&
    runRequestBuild.request !== null &&
    state.processingBlock === null &&
    isOnlineForIngest && !isCheckingConnection

  const handleStartProcessing = useCallback(() => {
    if (!canStartProcessing) return
    startProcessing()
  }, [canStartProcessing, startProcessing])

  // Contextual warnings
  const warnings = useMemo(() => {
    const result: string[] = []

    // Large files
    selectedQueueItems.forEach((item) => {
      if (item.fileSize > LARGE_FILE_THRESHOLD) {
        const name = item.fileName ?? item.url ?? item.id
        const size = formatFileSize(item.fileSize)
        result.push(
          qi("review.warnLargeFile", "{{name}} is {{size}} -- upload may take a moment", {
            name,
            size,
          })
        )
      }
    })

    // Long estimated time
    if (totalEstimatedSeconds > LONG_TIME_THRESHOLD) {
      result.push(
        qi("review.warnLongTime", "Processing may take a while ({{time}})", {
          time: estimatedTimeLabel,
        })
      )
    }

    // Large batch
    if (selectedQueueItems.length > LARGE_BATCH_THRESHOLD) {
      result.push(
        qi(
          "review.warnLargeBatch",
          "{{count}} items queued -- consider processing in smaller batches for better feedback",
          { count: selectedQueueItems.length }
        )
      )
    }

    return result
  }, [selectedQueueItems, totalEstimatedSeconds, estimatedTimeLabel, qi])

  // Item display name
  const getItemLabel = useCallback((item: WizardQueueItem): string => {
    if (item.playlist?.title) {
      return item.playlist.ordinal
        ? `${item.playlist.ordinal}. ${item.playlist.title}`
        : item.playlist.title
    }
    if (item.conferenceOverride?.title) return item.conferenceOverride.title
    if (item.fileName) return item.fileName
    if (item.url) {
      // Truncate long URLs for display
      const maxLen = 40
      return item.url.length > maxLen ? item.url.slice(0, maxLen) + "..." : item.url
    }
    return item.id
  }, [])

  return (
    <div className="flex h-full flex-col">
      {/* Summary header */}
      <div className="border-b border-border px-4 py-4 text-center sm:px-6">
        <h2 className="text-lg font-semibold text-text">
          {qi("review.title", "Ready to Process")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {qi("review.summary", "{{count}} items | {{preset}} preset | {{time}} estimated", {
            count: selectedQueueItems.length,
            preset: presetLabel,
            time: estimatedTimeLabel,
          })}
        </p>
      </div>

      {/* Scrollable item list */}
      <div className="flex-1 overflow-y-auto px-4 py-3 sm:px-6">
        {conferenceBatchMetadata && (
          <div
            className="mb-3 rounded-md border border-border bg-surface px-3 py-2 text-sm"
            aria-label="Conference batch review"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-text">
                {conferenceBatchMetadata.collectionName || "Conference batch"}
              </span>
              <span className="text-text-muted">{selectedQueueItems.length} selected</span>
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-muted">
              {conferenceBatchMetadata.conferenceName && (
                <span>{conferenceBatchMetadata.conferenceName}</span>
              )}
              {conferenceBatchMetadata.eventYear && (
                <span>{conferenceBatchMetadata.eventYear}</span>
              )}
              {conferenceBatchMetadata.eventDate && (
                <span>{conferenceBatchMetadata.eventDate}</span>
              )}
              {conferenceBatchMetadata.sharedTags.length > 0 && (
                <span>{conferenceBatchMetadata.sharedTags.join(", ")}</span>
              )}
            </div>
          </div>
        )}
        {queueItems.some((item) => item.playlist) && (
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <label className="text-xs text-text-muted">
              <span className="sr-only">Filter review items</span>
              <select
                aria-label="Filter review items"
                value={filter}
                onChange={(event) => setFilter(event.target.value as ReviewFilter)}
                className="rounded border border-border bg-surface px-2 py-1 text-xs"
              >
                <option value="selected">Selected</option>
                <option value="duplicates">Duplicates</option>
                <option value="policy">Policy chosen</option>
              </select>
            </label>
            <span className="text-xs text-text-muted" role="status" aria-live="polite">
              Showing {filteredReviewItems.length} of {queueItems.length} review items
            </span>
          </div>
        )}
        <div
          ref={reviewListRef}
          className="max-h-80 overflow-y-auto rounded-lg border border-border bg-surface2"
          role="list"
          aria-label={qi("review.itemList.ariaLabel", "Items to process")}
        >
          <div className="relative w-full" style={{ height: reviewVirtualizer.getTotalSize() }}>
            {reviewVirtualItems.map((virtualRow) => {
              const item = filteredReviewItems[virtualRow.index]
              if (!item) return null
              const IconComponent = TYPE_ICONS[item.detectedType] ?? File
              const ops = getOperationDescription(item.detectedType, selectedPreset, presetConfig)
              const label = getItemLabel(item)

              return (
                <div
                  key={virtualRow.key}
                  ref={(element) => {
                    const id = getQueueItemOccurrenceId(item)
                    if (element) {
                      reviewRowRefs.current.set(id, element)
                      reviewVirtualizer.measureElement(element)
                    } else {
                      reviewRowRefs.current.delete(id)
                    }
                  }}
                  role="listitem"
                  tabIndex={activeReviewId === getQueueItemOccurrenceId(item) ? 0 : -1}
                  aria-setsize={filteredReviewItems.length}
                  aria-posinset={virtualRow.index + 1}
                  data-occurrence-id={getQueueItemOccurrenceId(item)}
                  data-index={virtualRow.index}
                  onFocusCapture={() => {
                    const id = getQueueItemOccurrenceId(item)
                    reviewListOwnsFocusRef.current = true
                    activeReviewRowRef.current = { id, index: virtualRow.index }
                    setActiveReviewId(id)
                  }}
                  onKeyDown={(event) => handleReviewRowKeyDown(event, virtualRow.index)}
                  className="absolute left-0 top-0 flex w-full items-center gap-3 border-b border-border px-3 py-2.5 text-sm"
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                >
                  <IconComponent
                    className="h-4 w-4 flex-shrink-0 text-text-muted"
                    aria-hidden="true"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-medium text-text">
                      {label}
                      {item.conferenceOverride?.speaker && (
                        <span className="ml-2 font-normal text-text-muted">
                          {item.conferenceOverride.speaker}
                        </span>
                      )}
                      {item.playlist?.playlistTitle && (
                        <span className="ml-2 font-normal text-text-muted">
                          {item.playlist.playlistTitle}
                        </span>
                      )}
                    </div>
                    {item.sourceRef?.kind === "materialized_playlist_item" && item.url && (
                      <details className="font-normal text-[11px] text-text-muted">
                        <summary>Source details</summary>
                        <span>{item.url}</span>
                      </details>
                    )}
                  </div>
                  <span className="flex-shrink-0 whitespace-nowrap text-xs text-text-muted">
                    {presetLabel} &middot; {ops}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {queueItems.some((item) => item.sourceRef?.kind === "materialized_playlist_item") && (
          <ItemMetadataTable mode="playlist" visibleItemIds={visibleReviewItemIds} />
        )}

        {/* Storage mode */}
        <p className="mt-3 text-xs text-text-muted">
          {qi("review.storage", "Storage: {{mode}}", { mode: storageMode })}
        </p>

        {!isOnlineForIngest && (
          <DesignSystemAlert
            variant="warning"
            icon={<AlertTriangle className="h-4 w-4" />}
            className="mt-3"
            title={qi("wizard.offline.title", "Server offline")}
            action={
              onRetryConnection
                ? {
                    label: isCheckingConnection
                      ? qi("wizard.offline.checking", "Checking...")
                      : qi("wizard.offline.retry", "Retry connection"),
                    onClick: onRetryConnection,
                    loading: isCheckingConnection,
                    disabled: isCheckingConnection,
                  }
                : undefined
            }
          >
            {connectionRecoveryMessage ||
              qi(
                "wizard.offline.description",
                "Reconnect to your tldw server before processing. You can go back and keep editing the queue."
              )}
          </DesignSystemAlert>
        )}

        {materializationExpired && (
          <DesignSystemAlert
            variant="error"
            className="mt-3"
            title="This staged playlist expired. Inspect it again before processing."
          />
        )}

        {reviewRequired && (
          <DesignSystemAlert
            variant="warning"
            className="mt-3"
            title="Review duplicate actions and fix invalid metadata changes before processing."
          />
        )}

        {exceedsRunInputLimit && (
          <DesignSystemAlert
            variant="error"
            className="mt-3"
            title="Too many items selected. Select no more than 500 items before processing."
          />
        )}

        {/* Contextual warnings */}
        {warnings.length > 0 && (
          <div className="mt-3 space-y-2" role="alert">
            {warnings.map((warning, idx) => (
              <div
                key={idx}
                className="flex items-start gap-2 rounded-md bg-warn/10 px-3 py-2 text-xs text-warn"
              >
                <AlertTriangle
                  className="mt-0.5 h-3.5 w-3.5 flex-shrink-0"
                  aria-hidden="true"
                />
                <span>{warning}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer navigation */}
      <div className="flex items-center justify-between border-t border-border px-4 py-3 sm:px-6">
        <button
          type="button"
          onClick={goBack}
          className="inline-flex items-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium text-text-muted transition-colors hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
          aria-label={qi("review.backAriaLabel", "Back to Settings")}
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          {qi("review.backButton", "Back to Settings")}
        </button>

        <button
          type="button"
          onClick={handleStartProcessing}
          disabled={!canStartProcessing}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={qi("review.startAriaLabel", "Start processing")}
        >
          <Play className="h-4 w-4" aria-hidden="true" />
          {qi("review.startButton", "Start Processing")}
        </button>
      </div>
    </div>
  )
}

export default ReviewStep
