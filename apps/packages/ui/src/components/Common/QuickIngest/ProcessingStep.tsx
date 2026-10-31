import React, { useCallback, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
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

type FailedProcessingItem = {
  id: string
  label: string
  sourceUrl?: string
  fileName?: string
  error?: string
}

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

// ---------------------------------------------------------------------------
// ItemRow
// ---------------------------------------------------------------------------

type ItemRowProps = {
  item: WizardQueueItem
  progress: ItemProgress
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  onCancel: (id: string) => void
}

const ItemRow: React.FC<ItemRowProps> = ({ item, progress, qi, onCancel }) => {
  const IconComponent = TYPE_ICON_MAP[item.detectedType] || File
  const displayName = item.fileName || item.url || item.id
  const isTerminal = TERMINAL_STATUSES.has(progress.status)
  const isActive =
    progress.status !== "queued" &&
    progress.status !== "complete" &&
    progress.status !== "failed" &&
    progress.status !== "cancelled"

  const statusLabel = useMemo(() => {
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
  }, [progress.status, qi])

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

        {isActive && (
          <div
            role="progressbar"
            aria-label={qi("processing.itemProgress", "Processing {{name}}", {
              name: displayName,
            })}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress.progressPercent > 0 ? progress.progressPercent : undefined}
            className="h-1.5 overflow-hidden rounded-full bg-surface2"
          >
            <div
              className={
                progress.progressPercent > 0
                  ? "h-full rounded-full bg-primary transition-all duration-300"
                  : "h-full animate-pulse rounded-full bg-primary/30 motion-reduce:animate-none"
              }
              style={
                progress.progressPercent > 0
                  ? { width: `${progress.progressPercent}%` }
                  : undefined
              }
            />
          </div>
        )}

        {/* Error message */}
        {progress.status === "failed" && progress.error && (
          <p className="mt-0.5 text-xs text-danger">{progress.error}</p>
        )}
      </div>

      {/* Cancel button */}
      {!isTerminal && (
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
  onMinimize?: () => void
}

export const ProcessingStep: React.FC<ProcessingStepProps> = ({ onCancelAll, onMinimize }) => {
  const { t } = useTranslation(["option"])
  const { state, cancelProcessing, cancelItem, minimize } = useIngestWizard()
  const { processingState, queueItems } = state
  const tracking = useQuickIngestSessionStore((store) => store.session?.tracking)
  const [failedExportNotice, setFailedExportNotice] = useState<string | null>(null)

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

  const finishedCount = counts.completed + counts.failed + counts.cancelled
  const totalCount = processingState.perItemProgress.length
  const finishedPercent =
    totalCount > 0 ? Math.round((finishedCount / totalCount) * 100) : 0

  const handleCancelAll = useCallback(() => {
    if (onCancelAll) {
      onCancelAll()
      return
    }
    cancelProcessing()
  }, [cancelProcessing, onCancelAll])

  const handleMinimize = useCallback(() => {
    minimize()
    onMinimize?.()
  }, [minimize, onMinimize])

  const handleCancelItem = useCallback(
    (id: string) => {
      cancelItem(id)
    },
    [cancelItem]
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
            {qi("processing.finishedCount", "{{done}}/{{total}} finished", {
              done: finishedCount,
              total: totalCount,
            })}
          </span>
        )}
      </div>

      {/* Confirmed finished-item count */}
      <div
        role="progressbar"
        aria-label={qi("processing.finishedItems", "Finished items")}
        aria-valuemin={0}
        aria-valuemax={totalCount || 1}
        aria-valuenow={finishedCount}
        className="h-2 w-full overflow-hidden rounded-full bg-surface2"
      >
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            processingState.status === "cancelled"
              ? "bg-text-muted"
              : processingState.status === "error"
                ? "bg-danger"
                : "bg-primary"
          }`}
          style={{ width: `${finishedPercent}%` }}
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
                "processing.banner.waitingTitle",
                "Waiting for server results"
              )}
            </p>
            <p className="mt-0.5 text-xs text-text-muted">
              {qi(
                "processing.banner.waitingDescription",
                "Elapsed time is shown below. Completion time depends on the content and server."
              )}
            </p>
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
                "processing.banner.stillWaiting",
                "Still waiting for server results. You can leave this running and check later."
              )}
            </p>
          </div>
        )}

      {/* Item list */}
      <div className="flex max-h-[50vh] flex-col gap-2 overflow-y-auto" role="list">
        {processingState.perItemProgress.map((progress) => {
          const queueItem = queueItemMap.get(progress.id)
          if (!queueItem) return null

          return (
            <div key={progress.id} role="listitem">
              <ItemRow
                item={queueItem}
                progress={progress}
                qi={qi}
                onCancel={handleCancelItem}
              />
            </div>
          )
        })}
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
