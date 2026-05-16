import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import {
  Check,
  Circle,
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Ordered stages for per-item multi-stage progress indicator.
 */
const PROCESSING_STAGES = ["uploading", "processing", "analyzing", "storing"] as const
type ProcessingStage = (typeof PROCESSING_STAGES)[number]

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

/**
 * Determine the visual state of each stage dot relative to the item's current status.
 */
const getStageState = (
  stage: ProcessingStage,
  itemStatus: ItemProgressStatus
): "done" | "active" | "pending" | "failed" => {
  const stageIndex = PROCESSING_STAGES.indexOf(stage)
  const activeIndex = PROCESSING_STAGES.indexOf(itemStatus as ProcessingStage)

  if (itemStatus === "complete") return "done"
  if (itemStatus === "failed") {
    // Stages before the failure point are done; the failure point is failed; rest pending
    if (activeIndex >= 0) {
      if (stageIndex < activeIndex) return "done"
      if (stageIndex === activeIndex) return "failed"
      return "pending"
    }
    // If status doesn't map to a stage (e.g. failed during queued), mark first as failed
    return stageIndex === 0 ? "failed" : "pending"
  }
  if (itemStatus === "cancelled" || itemStatus === "queued") return "pending"

  // Active processing: check relative position
  if (activeIndex < 0) return "pending"
  if (stageIndex < activeIndex) return "done"
  if (stageIndex === activeIndex) return "active"
  return "pending"
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

type StageIndicatorProps = {
  stage: ProcessingStage
  visualState: "done" | "active" | "pending" | "failed"
  label: string
}

const StageIndicator: React.FC<StageIndicatorProps> = ({
  stage: _stage,
  visualState,
  label,
}) => {
  return (
    <div className="flex flex-col items-center gap-0.5" title={label}>
      <span className="flex h-4 w-4 items-center justify-center">
        {visualState === "done" ? (
          <Check className="h-3.5 w-3.5 text-primary" strokeWidth={2.5} aria-hidden="true" />
        ) : visualState === "active" ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" aria-hidden="true" />
        ) : visualState === "failed" ? (
          <X className="h-3.5 w-3.5 text-danger" strokeWidth={2.5} aria-hidden="true" />
        ) : (
          <Circle className="h-2.5 w-2.5 text-text-muted" aria-hidden="true" />
        )}
      </span>
      <span
        className={`text-[9px] leading-none ${
          visualState === "active"
            ? "font-medium text-primary"
            : visualState === "done"
              ? "text-text-muted"
              : visualState === "failed"
                ? "text-danger"
                : "text-text-muted opacity-50"
        }`}
      >
        {label}
      </span>
    </div>
  )
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

  const stageLabels: Record<ProcessingStage, string> = useMemo(
    () => ({
      uploading: qi("processing.stage.upload", "Upload"),
      processing: qi("processing.stage.process", "Process"),
      analyzing: qi("processing.stage.analyze", "Analyze"),
      storing: qi("processing.stage.store", "Store"),
    }),
    [qi]
  )

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

        {/* Multi-stage progress indicator */}
        {progress.status !== "queued" && (
          <div className="flex items-center gap-1">
            {/* Stage dots with connectors */}
            <div className="flex items-center gap-0">
              {PROCESSING_STAGES.map((stage, idx) => {
                const state = getStageState(stage, progress.status)
                return (
                  <React.Fragment key={stage}>
                    {idx > 0 && (
                      <div
                        className={`mx-0.5 h-px w-3 sm:w-5 ${
                          state === "done" || state === "active"
                            ? "bg-primary"
                            : state === "failed"
                              ? "bg-danger"
                              : "bg-border"
                        }`}
                        aria-hidden="true"
                      />
                    )}
                    <StageIndicator
                      stage={stage}
                      visualState={state}
                      label={stageLabels[stage]}
                    />
                  </React.Fragment>
                )
              })}
            </div>

            {/* Progress bar for active items */}
            {isActive && (
              <div className="ml-2 hidden h-1.5 flex-1 overflow-hidden rounded-full bg-surface2 sm:block">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-300"
                  style={{ width: `${progress.progressPercent}%` }}
                />
              </div>
            )}
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

export const ProcessingStep: React.FC = () => {
  const { t } = useTranslation(["option"])
  const { state, cancelProcessing, cancelItem, minimize } = useIngestWizard()
  const { processingState, queueItems } = state

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
    cancelProcessing()
  }, [cancelProcessing])

  const handleMinimize = useCallback(() => {
    minimize()
  }, [minimize])

  const handleCancelItem = useCallback(
    (id: string) => {
      cancelItem(id)
    },
    [cancelItem]
  )

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
            <p className="mt-0.5 text-xs text-text-muted">
              {qi(
                "processing.banner.subtitle",
                "Processing and indexing content"
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
                "processing.banner.timeout",
                "Processing is taking longer than usual. Your file will appear when ready."
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
