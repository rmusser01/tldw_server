import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useVirtualizer } from "@tanstack/react-virtual"
import {
  Check,
  Loader2,
  X,
  Minimize2,
  FileText,
  Video,
  Music,
  Image,
  Globe,
  BookOpen,
  FileQuestion,
  File,
  AlertTriangle,
} from "lucide-react"
import type { ItemProgress, ItemProgressStatus, WizardQueueItem } from "./types"
import { useIngestWizard } from "./IngestWizardContext"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { QUICK_INGEST_ACCEPT_STRING } from "./constants"
import { validateQuickIngestFile } from "./QueueTab/FileDropZone"
import { submitQuickIngestBatch } from "@/services/tldw/quick-ingest-batch"
import { readQuickIngestFileBytes } from "./file-bytes"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Map of detected media types to lucide icon components.
 */
const TYPE_ICON_MAP: Record<string, React.ElementType> = {
  audio: Music,
  video: Video,
  document: FileText,
  pdf: FileText,
  ebook: BookOpen,
  image: Image,
  web: Globe,
  unknown: FileQuestion,
}

/**
 * Terminal statuses where the item is no longer actively processing.
 */
const TERMINAL_STATUSES = new Set<ItemProgressStatus>([
  "complete",
  "failed",
  "cancelled",
])

const LIST_VIRTUALIZATION_THRESHOLD = 100

type FailedProcessingItem = {
  id: string
  label: string
  sourceUrl?: string
  fileName?: string
  error?: string
}

type LifecycleGroup = "active" | "attention" | "terminal"
type ProcessingFilter = "all" | LifecycleGroup

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Format seconds into MM:SS display string.
 */
const formatTime = (seconds: number): string => {
  if (seconds <= 0) return "0:00"
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

/**
 * Format estimated remaining time into a human-friendly string.
 */
const formatEstimated = (seconds: number): string => {
  if (seconds <= 0) return ""
  if (seconds < 60) return `~${Math.ceil(seconds)}s remaining`
  const m = Math.ceil(seconds / 60)
  return `~${m} min remaining`
}

const lifecycleGroup = (
  progress: ItemProgress,
  item?: WizardQueueItem
): LifecycleGroup => {
  if (
    progress.lifecycleState === "terminal" ||
    (!progress.lifecycleState && TERMINAL_STATUSES.has(progress.status))
  ) {
    return "terminal"
  }
  if (
    progress.lifecycleState === "status_unavailable" ||
    progress.lifecycleState === "cancellation_requested" ||
    (progress.lifecycleState === "awaiting_upload" && !item?.file)
  ) {
    return "attention"
  }
  return "active"
}

const itemDisplayName = (item: WizardQueueItem): string => {
  if (item.playlist?.ordinal && item.playlist.title) {
    return `${item.playlist.ordinal}. ${item.playlist.title}`
  }
  return item.playlist?.title || item.fileName || item.url || item.id
}

// ---------------------------------------------------------------------------
// ItemRow
// ---------------------------------------------------------------------------

type ItemRowProps = {
  item: WizardQueueItem
  progress: ItemProgress
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onCancel: (id: string) => void
  onCheckStatus: (id: string) => void
  onReconnect: () => void
  onReselectFile: (id: string, file: File) => void
  onRetryUpload: (id: string) => void
}

const ItemRow: React.FC<ItemRowProps> = ({
  item,
  progress,
  qi,
  onCancel,
  onCheckStatus,
  onReconnect,
  onReselectFile,
  onRetryUpload,
}) => {
  const IconComponent = TYPE_ICON_MAP[item.detectedType] || File
  const displayName = itemDisplayName(item)
  const group = lifecycleGroup(progress, item)
  const isTerminal = group === "terminal"
  const isActive = group === "active"
  const replacementInputRef = useRef<HTMLInputElement>(null)

  const statusLabel = useMemo(() => {
    switch (progress.lifecycleState) {
      case "staged":
      case "preparing":
        return qi("processing.status.preparing", "Preparing")
      case "awaiting_upload":
        return item.file
          ? qi("processing.status.awaitingUpload", "Awaiting upload")
          : qi("processing.status.fileReattachRequired", "File reattach required")
      case "submit_pending":
        return qi("processing.status.submitPending", "Submit pending")
      case "queued":
        return qi("processing.status.queued", "Queued")
      case "running":
        return qi("processing.status.running", "Running")
      case "cancellation_requested":
        return qi("processing.status.cancellationRequested", "Cancellation requested")
      case "status_unavailable":
        return qi("processing.status.statusUnavailable", "Status unavailable")
      case "terminal":
        switch (progress.terminalOutcome) {
          case "included_existing":
            return qi("processing.outcome.includedExisting", "Included existing")
          case "metadata_updated":
            return qi("processing.outcome.metadataUpdated", "Metadata updated")
          case "skipped_existing":
            return qi("processing.outcome.skippedExisting", "Skipped existing")
          case "submit_failed":
            return qi("processing.outcome.submitFailed", "Submit failed")
          case "processing_failed":
            return qi("processing.outcome.processingFailed", "Processing failed")
          case "metadata_update_failed":
            return qi("processing.outcome.metadataUpdateFailed", "Metadata update failed")
          case "cancelled":
            return qi("processing.status.cancelled", "Cancelled")
          default:
            return qi("processing.outcome.completed", "Completed")
        }
    }
    switch (progress.status) {
      case "queued":
        return qi("processing.status.queued", "Queued")
      case "uploading":
        return qi("processing.status.uploading", "Uploading")
      case "processing":
        return qi("processing.status.processing", "Processing")
      case "analyzing":
        return qi("processing.status.analyzing", "Analyzing")
      case "storing":
        return qi("processing.status.storing", "Storing")
      case "complete":
        return qi("processing.status.complete", "Complete")
      case "failed":
        return qi("processing.status.failed", "Failed")
      case "cancelled":
        return qi("processing.status.cancelled", "Cancelled")
      default:
        return ""
    }
  }, [item.file, progress.lifecycleState, progress.status, progress.terminalOutcome, qi])

  const handleCancel = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onCancel(item.id)
    },
    [item.id, onCancel]
  )

  return (
    <div
      className={`flex items-center gap-3 rounded-md border px-3 py-2 transition ${
        progress.status === "complete"
          ? "border-primary/30 bg-primary/5"
          : progress.status === "failed"
            ? "border-danger/30 bg-danger/5"
            : progress.status === "cancelled"
              ? "border-border bg-surface2/50 opacity-60"
              : "border-border"
      }`}
    >
      {/* Icon */}
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center">
        {progress.status === "complete" ? (
          <Check className="h-5 w-5 text-primary" strokeWidth={2.5} aria-hidden="true" />
        ) : progress.status === "failed" ? (
          <X className="h-5 w-5 text-danger" strokeWidth={2.5} aria-hidden="true" />
        ) : (
          <IconComponent className="h-5 w-5 text-text-muted" aria-hidden="true" />
        )}
      </div>

      {/* Content */}
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {/* Name + status line */}
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-sm font-medium text-text" title={displayName}>
            {displayName}
          </span>
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`whitespace-nowrap ${
                progress.status === "complete"
                  ? "text-primary"
                  : progress.status === "failed"
                    ? "text-danger"
                    : progress.status === "cancelled"
                      ? "text-text-muted"
                      : "text-text"
              }`}
            >
              {statusLabel}
            </span>
            {isActive && progress.progressPercent > 0 && (
              <span className="tabular-nums text-text-muted">
                {progress.progressPercent}%
              </span>
            )}
            {isActive && progress.estimatedRemaining > 0 && (
              <span className="hidden whitespace-nowrap text-text-muted sm:inline">
                {formatEstimated(progress.estimatedRemaining)}
              </span>
            )}
          </div>
        </div>

        {progress.currentStage && progress.currentStage !== statusLabel && (
          <p className="text-xs text-text-muted">{progress.currentStage}</p>
        )}

        {isActive && progress.progressPercent > 0 && (
          <div className="h-1.5 overflow-hidden rounded-full bg-surface2">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${progress.progressPercent}%` }}
            />
          </div>
        )}

        {/* Error message */}
        {progress.status === "failed" && progress.error && (
          <p className="mt-0.5 text-xs text-danger">{progress.error}</p>
        )}
      </div>

      {progress.lifecycleState === "status_unavailable" && (
        <div className="flex flex-shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={() => onCheckStatus(item.id)}
            className="rounded px-2 py-1 text-xs text-primary hover:bg-primary/10"
          >
            {qi("processing.checkAgain", "Check again")}
          </button>
          <button
            type="button"
            onClick={onReconnect}
            className="rounded px-2 py-1 text-xs text-primary hover:bg-primary/10"
          >
            {qi("processing.reconnect", "Reconnect")}
          </button>
        </div>
      )}

      {progress.lifecycleState === "awaiting_upload" && !item.file && (
        <div className="flex flex-shrink-0 items-center">
          <input
            ref={replacementInputRef}
            type="file"
            accept={QUICK_INGEST_ACCEPT_STRING}
            className="sr-only"
            aria-label={qi(
              "processing.replacementFileAria",
              "Replacement file for {{name}}",
              { name: displayName }
            )}
            onChange={(event) => {
              const file = event.currentTarget.files?.[0]
              if (file) onReselectFile(item.id, file)
              event.currentTarget.value = ""
            }}
          />
          <button
            type="button"
            className="rounded px-2 py-1 text-xs text-primary hover:bg-primary/10"
            aria-label={qi(
              "processing.reselectFileAria",
              "Reselect file for {{name}}",
              { name: displayName }
            )}
            onClick={() => replacementInputRef.current?.click()}
          >
            {qi("processing.reselectFile", "Reselect file")}
          </button>
        </div>
      )}

      {progress.lifecycleState === "awaiting_upload" && item.file && (
        <button
          type="button"
          className="flex-shrink-0 rounded px-2 py-1 text-xs text-primary hover:bg-primary/10"
          aria-label={qi(
            "processing.retryUploadAria",
            "Retry upload for {{name}}",
            { name: displayName }
          )}
          onClick={() => onRetryUpload(item.id)}
        >
          {qi("processing.retryUpload", "Retry upload")}
        </button>
      )}

      {/* Cancel button */}
      {!isTerminal && progress.lifecycleState !== "cancellation_requested" && (
        <button
          type="button"
          onClick={handleCancel}
          className="flex-shrink-0 rounded px-2 py-1 text-xs text-text-muted transition hover:bg-surface2 hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
          aria-label={qi("processing.cancelItem", "Cancel {{name}}", {
            name: displayName,
          })}
        >
          {qi("processing.cancel", "Cancel")}
        </button>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ProcessingStep (main component)
// ---------------------------------------------------------------------------

type ProcessingStepProps = {
  onCancelAll?: () => void
}

export const ProcessingStep: React.FC<ProcessingStepProps> = ({ onCancelAll }) => {
  const { t } = useTranslation(["option"])
  const {
    state,
    cancelProcessing,
    cancelItem,
    checkStatus,
    reconnect,
    minimize,
    updateQueueItems,
    updateItemProgress,
  } = useIngestWizard()
  const { processingState, queueItems } = state
  const tracking = useQuickIngestSessionStore((store) => store.session?.tracking)
  const [failedExportNotice, setFailedExportNotice] = useState<string | null>(null)
  const [processingFilter, setProcessingFilter] = useState<ProcessingFilter>("all")
  const [focusedProcessingIndex, setFocusedProcessingIndex] = useState(0)
  const processingListRef = useRef<HTMLDivElement>(null)
  const pendingProcessingFocusRef = useRef<number | null>(null)

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // Build a lookup map from queue items by ID for O(1) access
  const queueItemMap = useMemo(() => {
    const map = new Map<string, WizardQueueItem>()
    for (const item of queueItems) {
      map.set(item.id, item)
    }
    return map
  }, [queueItems])

  const lifecycleCounts = useMemo(() => {
    const result: Record<LifecycleGroup, number> = {
      active: 0,
      attention: 0,
      terminal: 0,
    }
    for (const progress of processingState.perItemProgress) {
      result[lifecycleGroup(progress, queueItemMap.get(progress.id))] += 1
    }
    return result
  }, [processingState.perItemProgress, queueItemMap])

  const filteredProgress = useMemo(
    () =>
      processingState.perItemProgress.filter(
        (progress) =>
          processingFilter === "all" ||
          lifecycleGroup(progress, queueItemMap.get(progress.id)) === processingFilter
      ),
    [processingFilter, processingState.perItemProgress, queueItemMap]
  )
  const usesVirtualProcessing =
    processingState.perItemProgress.length >= LIST_VIRTUALIZATION_THRESHOLD

  // TanStack Virtual exposes an imperative object that React Compiler skips.
  // eslint-disable-next-line react-hooks/incompatible-library
  const processingVirtualizer = useVirtualizer({
    count: usesVirtualProcessing ? filteredProgress.length : 0,
    getScrollElement: () => processingListRef.current,
    estimateSize: () => 76,
    overscan: 6,
    getItemKey: (index) => filteredProgress[index]?.id ?? index,
    measureElement: (element) => element?.getBoundingClientRect().height || 76,
  })
  const processingVirtualItems = processingVirtualizer.getVirtualItems()

  const focusProcessingIndex = useCallback(
    (requestedIndex: number) => {
      if (filteredProgress.length === 0) return
      const nextIndex = Math.max(
        0,
        Math.min(requestedIndex, filteredProgress.length - 1)
      )
      setFocusedProcessingIndex(nextIndex)
      const row = processingListRef.current?.querySelector<HTMLElement>(
        `[data-index="${nextIndex}"]`
      )
      if (row) {
        pendingProcessingFocusRef.current = null
        row.focus()
        return
      }
      pendingProcessingFocusRef.current = nextIndex
      processingVirtualizer.scrollToIndex(nextIndex)
    },
    [filteredProgress.length, processingVirtualizer]
  )

  useEffect(() => {
    const pendingIndex = pendingProcessingFocusRef.current
    if (pendingIndex == null) return
    const row = processingListRef.current?.querySelector<HTMLElement>(
      `[data-index="${pendingIndex}"]`
    )
    if (!row) return
    pendingProcessingFocusRef.current = null
    row.focus()
  }, [processingVirtualItems])

  useEffect(() => {
    if (filteredProgress.length === 0) {
      setFocusedProcessingIndex(0)
      return
    }
    if (focusedProcessingIndex >= filteredProgress.length) {
      setFocusedProcessingIndex(filteredProgress.length - 1)
    }
  }, [filteredProgress.length, focusedProcessingIndex])

  const workerProgressMessage = useMemo(
    () =>
      processingState.perItemProgress.find(
        (progress) =>
          lifecycleGroup(progress, queueItemMap.get(progress.id)) === "active" &&
          progress.currentStage
      )?.currentStage || null,
    [processingState.perItemProgress, queueItemMap]
  )

  const trackingSummary = useMemo(() => {
    if (!tracking) return null

    const plannedCount = tracking.plannedItemIds?.length ?? 0
    const jobCount = tracking.jobIds?.length ?? 0
    const batchCount = tracking.batchIds?.length ?? (tracking.batchId ? 1 : 0)
    const hasAnyTracking =
      Boolean(tracking.collectionId) ||
      plannedCount > 0 ||
      jobCount > 0 ||
      batchCount > 0 ||
      Boolean(tracking.sessionId)

    if (!hasAnyTracking) return null

    const modeLabel =
      tracking.durableMode === "durable_collection"
        ? qi("processing.tracking.durable", "Durable collection tracking")
        : tracking.durableMode === "degraded"
          ? qi("processing.tracking.degraded", "Local run tracking")
          : qi("processing.tracking.job", "Job tracking")

    const details: string[] = []
    if (tracking.collectionId) {
      details.push(
        qi("processing.tracking.collection", "Collection {{id}}", {
          id: tracking.collectionId,
        })
      )
    }
    if (plannedCount > 0) {
      details.push(
        plannedCount === 1
          ? qi("processing.tracking.plannedOne", "1 planned item")
          : qi("processing.tracking.plannedMany", "{{count}} planned items", {
              count: plannedCount,
            })
      )
    }
    if (jobCount > 0) {
      details.push(
        jobCount === 1
          ? qi("processing.tracking.jobOne", "1 job")
          : qi("processing.tracking.jobMany", "{{count}} jobs", {
              count: jobCount,
            })
      )
    } else if (batchCount > 0) {
      details.push(
        batchCount === 1
          ? qi("processing.tracking.batchOne", "1 batch")
          : qi("processing.tracking.batchMany", "{{count}} batches", {
              count: batchCount,
            })
      )
    }

    return { modeLabel, details }
  }, [qi, tracking])

  const failedItems = useMemo<FailedProcessingItem[]>(() => {
    return processingState.perItemProgress
      .filter((progress) => progress.status === "failed")
      .map((progress) => {
        const queueItem = queueItemMap.get(progress.id)
        const sourceUrl = queueItem?.url
        const fileName = queueItem?.fileName
        return {
          id: progress.id,
          label: sourceUrl || fileName || queueItem?.id || progress.id,
          sourceUrl,
          fileName,
          error: progress.error,
        }
      })
  }, [processingState.perItemProgress, queueItemMap])

  // Compute summary counts
  const counts = useMemo(() => {
    const result = { completed: 0, processing: 0, queued: 0, failed: 0, cancelled: 0 }
    for (const p of processingState.perItemProgress) {
      switch (p.status) {
        case "complete":
          result.completed++
          break
        case "failed":
          result.failed++
          break
        case "cancelled":
          result.cancelled++
          break
        case "queued":
          result.queued++
          break
        default:
          result.processing++
      }
    }
    return result
  }, [processingState.perItemProgress])

  // Overall progress
  const overallPercent = useMemo(() => {
    const items = processingState.perItemProgress
    if (items.length === 0) return 0
    const total = items.reduce((sum, p) => sum + p.progressPercent, 0)
    return Math.round(total / items.length)
  }, [processingState.perItemProgress])

  const handleCancelAll = useCallback(() => {
    if (onCancelAll) {
      onCancelAll()
      return
    }
    cancelProcessing()
  }, [cancelProcessing, onCancelAll])

  const handleMinimize = useCallback(() => {
    minimize()
  }, [minimize])

  const handleCancelItem = useCallback(
    (id: string) => {
      cancelItem(id)
    },
    [cancelItem]
  )

  const uploadFile = useCallback(
    (id: string, file: File) => {
      const validationError = validateQuickIngestFile(file)
      const currentProgress = processingState.perItemProgress.find(
        (progress) => progress.id === id
      )
      if (validationError) {
        updateItemProgress({
          id,
          attempt: currentProgress?.attempt,
          status: "queued",
          lifecycleState: "awaiting_upload",
          terminalOutcome: null,
          progressPercent: currentProgress?.progressPercent ?? 0,
          currentStage: validationError,
          estimatedRemaining: currentProgress?.estimatedRemaining ?? 0,
          error: validationError,
          retryable: true,
        })
        return
      }

      const sessionId = String(tracking?.sessionId || "").trim()
      const runId = String(tracking?.runId || "").trim()
      if (!sessionId || !runId) return
      void (async () => {
        try {
          const data = Array.from(
            new Uint8Array(await readQuickIngestFileBytes(file))
          )
          const response = await submitQuickIngestBatch({
            entries: [],
            files: [
              {
                id,
                name: file.name,
                type: file.type || undefined,
                data,
              },
            ],
            storeRemote: state.presetConfig.storeRemote,
            processOnly: !state.presetConfig.storeRemote,
            common: state.presetConfig.common,
            advancedValues: state.presetConfig.advancedValues,
            pendingRunRequest: {
              inputs: [
                {
                  inputKind: "file_stub",
                  occurrenceId: id,
                  attempt:
                    Number.isSafeInteger(currentProgress?.attempt) &&
                    Number(currentProgress?.attempt) > 0
                      ? Number(currentProgress?.attempt)
                      : 1,
                  name: file.name,
                  contentType: file.type || undefined,
                  sizeBytes: file.size,
                },
              ],
            },
            __quickIngestSessionId: sessionId,
            __quickIngestRunId: runId,
          })
          if (!response.accepted) {
            throw new Error(
              response.error || "The replacement file was not accepted."
            )
          }
          updateItemProgress({
            id,
            attempt: currentProgress?.attempt,
            status: "queued",
            lifecycleState: "queued",
            terminalOutcome: null,
            progressPercent: currentProgress?.progressPercent ?? 0,
            currentStage: qi("processing.status.queued", "Queued"),
            estimatedRemaining: 0,
            retryable: false,
          })
        } catch (error) {
          updateItemProgress({
            id,
            attempt: currentProgress?.attempt,
            status: "queued",
            lifecycleState: "awaiting_upload",
            terminalOutcome: null,
            progressPercent: currentProgress?.progressPercent ?? 0,
            currentStage:
              error instanceof Error
                ? error.message
                : qi(
                    "processing.status.fileUploadFailed",
                    "Replacement file upload failed. Try again."
                  ),
            estimatedRemaining: 0,
            error:
              error instanceof Error
                ? error.message
                : "Replacement file upload failed.",
            retryable: true,
          })
        }
      })()
    },
    [
      processingState.perItemProgress,
      qi,
      state.presetConfig,
      tracking,
      updateItemProgress,
    ]
  )

  const handleReselectFile = useCallback(
    (id: string, file: File) => {
      if (validateQuickIngestFile(file)) {
        uploadFile(id, file)
        return
      }
      const currentProgress = processingState.perItemProgress.find(
        (progress) => progress.id === id
      )
      updateQueueItems((items) =>
        items.map((item) =>
          item.id === id
            ? {
                ...item,
                file,
                fileName: file.name,
                fileSize: file.size,
                mimeType: file.type || undefined,
                validation: { valid: true },
              }
            : item
        )
      )
      updateItemProgress({
        id,
        attempt: currentProgress?.attempt,
        status: "queued",
        lifecycleState: "awaiting_upload",
        terminalOutcome: null,
        progressPercent: currentProgress?.progressPercent ?? 0,
        currentStage: qi(
          "processing.status.fileSelected",
          "File selected. Ready to upload."
        ),
        estimatedRemaining: currentProgress?.estimatedRemaining ?? 0,
        retryable: true,
      })
      uploadFile(id, file)
    },
    [
      processingState.perItemProgress,
      qi,
      updateItemProgress,
      updateQueueItems,
      uploadFile,
    ]
  )

  const handleRetryUpload = useCallback(
    (id: string) => {
      const file = queueItemMap.get(id)?.file
      if (file) uploadFile(id, file)
    },
    [queueItemMap, uploadFile]
  )

  const handleProcessingRowKeyDown = useCallback(
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
          nextIndex = filteredProgress.length - 1
          break
      }
      if (nextIndex == null) return
      event.preventDefault()
      focusProcessingIndex(nextIndex)
    },
    [filteredProgress.length, focusProcessingIndex]
  )

  const handleProcessingFilterChange = useCallback(
    (nextFilter: ProcessingFilter) => {
      setProcessingFilter(nextFilter)
      setFocusedProcessingIndex(0)
      pendingProcessingFocusRef.current = 0
      processingVirtualizer.scrollToIndex(0)
    },
    [processingVirtualizer]
  )

  const handleExportFailedItems = useCallback(async () => {
    if (failedItems.length === 0) {
      setFailedExportNotice(
        qi("processing.failedExportEmpty", "No failed items to export.")
      )
      return
    }

    const text = failedItems
      .map((item, index) => {
        const lines = [`#${index + 1}`]
        if (item.sourceUrl) {
          lines.push(`URL: ${item.sourceUrl}`)
        } else if (item.fileName) {
          lines.push(`File: ${item.fileName}`)
        } else {
          lines.push(`Item: ${item.label}`)
        }
        lines.push(`ID: ${item.id}`)
        if (item.error) {
          lines.push(`Error: ${item.error}`)
        }
        return lines.join("\n")
      })
      .join("\n\n")

    try {
      if (
        typeof navigator === "undefined" ||
        typeof navigator.clipboard?.writeText !== "function"
      ) {
        throw new Error("Clipboard unavailable")
      }
      await navigator.clipboard.writeText(text)
      setFailedExportNotice(
        qi("processing.failedExportCopied", "Failed list copied.")
      )
      return
    } catch {
      if (typeof document !== "undefined" && typeof URL !== "undefined") {
        const blob = new Blob([text], { type: "text/plain" })
        const url = URL.createObjectURL(blob)
        const anchor = document.createElement("a")
        anchor.href = url
        anchor.download = "quick-ingest-failed-items.txt"
        anchor.click()
        URL.revokeObjectURL(url)
        setFailedExportNotice(
          qi("processing.failedExportDownloaded", "Failed list downloaded.")
        )
      }
    }
  }, [failedItems, qi])

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text">
          {qi("processing.title", "Processing")}
        </h3>
        {processingState.status === "running" && (
          <span className="flex items-center gap-1.5 text-xs text-primary">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            {overallPercent}%
          </span>
        )}
      </div>

      {/* Overall progress bar */}
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface2">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            processingState.status === "cancelled"
              ? "bg-text-muted"
              : processingState.status === "error"
                ? "bg-danger"
                : "bg-primary"
          }`}
          style={{ width: `${overallPercent}%` }}
        />
      </div>

      {/* Durable run tracking */}
      {trackingSummary && (
        <div
          className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text-muted"
          data-testid="quick-ingest-run-tracking"
          role="status"
          aria-live="polite"
        >
          <span className="font-medium text-text">{trackingSummary.modeLabel}</span>
          {trackingSummary.details.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {trackingSummary.details.map((detail) => (
                <span
                  key={detail}
                  className="rounded border border-border bg-surface px-2 py-0.5"
                >
                  {detail}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Descriptive processing banner */}
      {processingState.status === "running" && counts.processing > 0 && (
        <div
          className="flex items-start gap-2.5 rounded-md border border-primary/20 bg-primary/5 px-3 py-2.5"
          role="status"
          aria-live="polite"
        >
          <Loader2
            className="mt-0.5 h-4 w-4 flex-shrink-0 animate-spin text-primary"
            aria-hidden="true"
          />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-text">
              {qi(
                "processing.banner.title",
                "Processing content... This may take a few minutes for large files."
              )}
            </p>
            {workerProgressMessage && (
              <p className="mt-0.5 text-xs text-text-muted">{workerProgressMessage}</p>
            )}
          </div>
        </div>
      )}

      {/* Timeout warning banner */}
      {processingState.status === "running" &&
        processingState.elapsed >= 300 &&
        counts.processing > 0 && (
          <div
            className="flex items-start gap-2.5 rounded-md border border-warn/30 bg-warn/5 px-3 py-2.5"
            role="alert"
          >
            <AlertTriangle
              className="mt-0.5 h-4 w-4 flex-shrink-0 text-warn"
              aria-hidden="true"
            />
            <p className="text-xs text-text-muted">
              {qi(
                "processing.banner.timeout",
                "Processing is taking longer than usual. Your file will appear when ready."
              )}
            </p>
          </div>
        )}

      <div className="flex flex-wrap items-center gap-2">
        <label className="text-xs text-text-muted" htmlFor="quick-ingest-processing-filter">
          {qi("processing.filter.label", "Show")}
        </label>
        <select
          id="quick-ingest-processing-filter"
          aria-label={qi("processing.filter.aria", "Filter processing items")}
          className="rounded border border-border bg-surface px-2 py-1 text-xs text-text"
          value={processingFilter}
          onChange={(event) =>
            handleProcessingFilterChange(event.target.value as ProcessingFilter)
          }
        >
          <option value="all">
            {qi("processing.filter.all", "All")} ({processingState.perItemProgress.length})
          </option>
          <option value="active">
            {qi("processing.filter.active", "Active")} ({lifecycleCounts.active})
          </option>
          <option value="attention">
            {qi("processing.filter.attention", "Needs attention")} ({lifecycleCounts.attention})
          </option>
          <option value="terminal">
            {qi("processing.filter.terminal", "Terminal")} ({lifecycleCounts.terminal})
          </option>
        </select>
      </div>

      {/* Item list */}
      <div
        ref={processingListRef}
        className="max-h-[50vh] overflow-y-auto"
        role="list"
        aria-label={qi("processing.items.aria", "Processing items")}
      >
        {!usesVirtualProcessing && (
          <div className="space-y-2">
            {filteredProgress.map((progress, index) => {
              const queueItem = queueItemMap.get(progress.id)
              if (!queueItem) return null
              const group = lifecycleGroup(progress, queueItem)
              return (
                <div
                  key={progress.id}
                  role="listitem"
                  tabIndex={0}
                  aria-setsize={filteredProgress.length}
                  aria-posinset={index + 1}
                  data-lifecycle-group={group}
                  data-index={index}
                >
                  <ItemRow
                    item={queueItem}
                    progress={progress}
                    qi={qi}
                    onCancel={handleCancelItem}
                    onCheckStatus={checkStatus}
                    onReconnect={reconnect}
                    onReselectFile={handleReselectFile}
                    onRetryUpload={handleRetryUpload}
                  />
                </div>
              )
            })}
          </div>
        )}
        {usesVirtualProcessing && <div
          className="relative w-full"
          style={{ height: processingVirtualizer.getTotalSize() }}
        >
          {processingVirtualItems.map((virtualRow) => {
            const progress = filteredProgress[virtualRow.index]
            const queueItem = progress ? queueItemMap.get(progress.id) : undefined
            if (!progress || !queueItem) return null
            const group = lifecycleGroup(progress, queueItem)

            return (
              <div
                key={virtualRow.key}
                ref={processingVirtualizer.measureElement}
                role="listitem"
                tabIndex={virtualRow.index === focusedProcessingIndex ? 0 : -1}
                aria-setsize={filteredProgress.length}
                aria-posinset={virtualRow.index + 1}
                data-lifecycle-group={group}
                data-index={virtualRow.index}
                onFocus={() => setFocusedProcessingIndex(virtualRow.index)}
                onKeyDown={(event) =>
                  handleProcessingRowKeyDown(event, virtualRow.index)
                }
                className="absolute left-0 top-0 w-full pb-2"
                style={{ transform: `translateY(${virtualRow.start}px)` }}
              >
                <ItemRow
                  item={queueItem}
                  progress={progress}
                  qi={qi}
                  onCancel={handleCancelItem}
                  onCheckStatus={checkStatus}
                  onReconnect={reconnect}
                  onReselectFile={handleReselectFile}
                  onRetryUpload={handleRetryUpload}
                />
              </div>
            )
          })}
        </div>}
      </div>

      {/* Summary bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-surface2 px-3 py-2 text-xs text-text-muted">
        <div className="flex flex-wrap items-center gap-3">
          <span>
            {qi("processing.completed", "Completed")}: {counts.completed}
          </span>
          <span className="text-border">|</span>
          <span>
            {qi("processing.inProgress", "Processing")}: {counts.processing}
          </span>
          <span className="text-border">|</span>
          <span>
            {qi("processing.queued", "Queued")}: {counts.queued}
          </span>
          {counts.failed > 0 && (
            <>
              <span className="text-border">|</span>
              <span className="text-danger">
                {qi("processing.failed", "Failed")}: {counts.failed}
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="tabular-nums">
            {qi("processing.elapsed", "Elapsed")}: {formatTime(processingState.elapsed)}
          </span>
          {processingState.estimatedRemaining > 0 && (
            <>
              <span className="text-border">|</span>
              <span className="tabular-nums">
                {qi("processing.estRemaining", "Est. remaining")}:{" "}
                ~{formatTime(processingState.estimatedRemaining)}
              </span>
            </>
          )}
        </div>
      </div>

      {failedItems.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-danger/20 bg-danger/5 px-3 py-2 text-xs">
          <div className="min-w-0 text-danger">
            <span className="font-medium">
              {failedItems.length === 1
                ? qi("processing.failedItemsOne", "1 failed item")
                : qi("processing.failedItemsMany", "{{count}} failed items", {
                    count: failedItems.length,
                  })}
            </span>
            {failedExportNotice && (
              <span className="ml-2 text-text-muted">{failedExportNotice}</span>
            )}
          </div>
          <button
            type="button"
            onClick={() => {
              void handleExportFailedItems()
            }}
            className="rounded-md border border-danger/30 px-3 py-1.5 text-xs font-medium text-danger transition hover:bg-danger/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
            aria-label={qi(
              "processing.exportFailedListAria",
              "Export failed items list"
            )}
          >
            {qi("processing.exportFailedList", "Export failed list")}
          </button>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center justify-end gap-2">
        {(processingState.status === "running" ||
          processingState.status === "idle") && (
          <button
            type="button"
            onClick={handleCancelAll}
            className="rounded-md border border-danger/30 px-3 py-1.5 text-xs font-medium text-danger transition hover:bg-danger/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
          >
            {qi("processing.cancelAll", "Cancel All")}
          </button>
        )}
        {processingState.status === "running" && (
          <button
            type="button"
            onClick={handleMinimize}
            className="flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text transition hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
          >
            <Minimize2 className="h-3.5 w-3.5" aria-hidden="true" />
            {qi("processing.minimize", "Minimize to Background")}
          </button>
        )}
      </div>
    </div>
  )
}

export default ProcessingStep
