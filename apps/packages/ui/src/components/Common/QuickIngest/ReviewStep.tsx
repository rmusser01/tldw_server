import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
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
import { useIngestWizard } from "./IngestWizardContext"
import { estimateTotalSeconds, formatEstimate } from "./timeEstimation"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const LARGE_FILE_THRESHOLD = 50 * 1024 * 1024 // 50 MB
const LONG_TIME_THRESHOLD = 15 * 60 // 15 minutes in seconds
const LARGE_BATCH_THRESHOLD = 5

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
  const { state, goBack, goNext, startProcessing } = useIngestWizard()

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  const { queueItems, selectedPreset, presetConfig } = state

  // Compute total estimated time
  const totalEstimatedSeconds = useMemo(
    () => estimateTotalSeconds(queueItems, selectedPreset),
    [queueItems, selectedPreset]
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
    () => queueItems.filter((item) => item.validation.valid).length,
    [queueItems]
  )
  const canStartProcessing =
    validItemCount > 0 && isOnlineForIngest && !isCheckingConnection

  const handleStartProcessing = useCallback(() => {
    if (!canStartProcessing) return
    startProcessing()
    goNext()
  }, [canStartProcessing, goNext, startProcessing])

  // Contextual warnings
  const warnings = useMemo(() => {
    const result: string[] = []

    // Large files
    queueItems.forEach((item) => {
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
        qi("review.warnLongTime", "Processing may take a while (~{{time}})", {
          time: estimatedTimeLabel,
        })
      )
    }

    // Large batch
    if (queueItems.length > LARGE_BATCH_THRESHOLD) {
      result.push(
        qi(
          "review.warnLargeBatch",
          "{{count}} items queued -- consider processing in smaller batches for better feedback",
          { count: queueItems.length }
        )
      )
    }

    return result
  }, [queueItems, totalEstimatedSeconds, estimatedTimeLabel, qi])

  // Item display name
  const getItemLabel = useCallback((item: WizardQueueItem): string => {
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
            count: queueItems.length,
            preset: presetLabel,
            time: estimatedTimeLabel,
          })}
        </p>
      </div>

      {/* Scrollable item list */}
      <div className="flex-1 overflow-y-auto px-4 py-3 sm:px-6">
        <ul
          className="divide-y divide-border rounded-lg border border-border bg-surface2"
          role="list"
          aria-label={qi("review.itemList.ariaLabel", "Items to process")}
        >
          {queueItems.map((item) => {
            const IconComponent = TYPE_ICONS[item.detectedType] ?? File
            const ops = getOperationDescription(item.detectedType, selectedPreset, presetConfig)
            const label = getItemLabel(item)

            return (
              <li
                key={item.id}
                className="flex items-center gap-3 px-3 py-2.5 text-sm"
              >
                <IconComponent
                  className="h-4 w-4 flex-shrink-0 text-text-muted"
                  aria-hidden="true"
                />
                <span className="min-w-0 flex-1 truncate font-medium text-text" title={item.fileName ?? item.url}>
                  {label}
                </span>
                <span className="flex-shrink-0 whitespace-nowrap text-xs text-text-muted">
                  {presetLabel} &middot; {ops}
                </span>
              </li>
            )
          })}
        </ul>

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
            {
              connectionRecoveryMessage ||
              qi(
                "wizard.offline.description",
                "Reconnect to your tldw server before processing. You can go back and keep editing the queue."
              )
            }
          </DesignSystemAlert>
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
