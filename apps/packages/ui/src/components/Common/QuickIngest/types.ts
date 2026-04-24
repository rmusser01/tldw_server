/**
 * Represents a status summary for Quick Ingest items.
 * @property label - Display text for the status.
 * @property color - Color identifier for the status badge.
 * @property reason - Optional explanation for the status (e.g., error details).
 */
export type StatusSummary = {
  label: string
  color: string
  reason?: string
}

export type QueuedFileStub = {
  id: string
  name: string
  size: number
  type?: string
}

export type ResultItem = {
  id: string
  status: "ok" | "error"
  outcome?: ResultOutcome
  url?: string
  fileName?: string
  type: string
  data?: unknown
  error?: string
  /** Whether the original file was persisted during this ingest run. */
  persisted?: boolean
}

export type ResultOutcome =
  | "ingested"
  | "processed"
  | "skipped"
  | "failed"
  | "cancelled"

export type ResultItemWithMediaId = ResultItem & {
  mediaId: string | number | null
  title?: string | null
}

export type ResultSummary = {
  successCount: number
  failCount: number
  cancelledCount: number
}

export type ResultFilters = {
  ALL: string
  ERROR: string
  SUCCESS: string
}

export type ResultsFilter = ResultFilters[keyof ResultFilters]

/**
 * Tab identifiers for the Quick Ingest modal.
 */
export type QuickIngestTab = "queue" | "options" | "results"

/**
 * Common processing options shared across all media types.
 */
export type CommonOptions = {
  perform_analysis: boolean
  perform_chunking: boolean
  overwrite_existing: boolean
}

/**
 * Type-specific default options applied per media type.
 */
export type TypeDefaults = {
  audio?: { language?: string; diarize?: boolean }
  document?: { ocr?: boolean }
  video?: { captions?: boolean }
}

/**
 * Preset identifiers for Quick Ingest option configurations.
 */
export type IngestPreset = "quick" | "standard" | "deep" | "custom"

/**
 * Configuration for a preset, defining all option values.
 */
export type PresetConfig = {
  common: CommonOptions
  storeRemote: boolean
  reviewBeforeStorage: boolean
  typeDefaults: TypeDefaults
  advancedValues?: Record<string, unknown>
}

/**
 * Chunking template options for ingest.
 */
export type ChunkingTemplateOptions = {
  /** Name of the selected chunking template */
  templateName?: string
  /** Whether to automatically detect and apply templates based on content */
  autoApply?: boolean
}

/**
 * Badge state for tab indicators.
 */
export type TabBadgeState = {
  /** Number of items in queue (for Queue tab) */
  queueCount: number
  /** Whether options have been modified from defaults (for Options tab) */
  optionsModified: boolean
  /** Whether processing is currently running (for Results tab) */
  isProcessing: boolean
  /** Whether the latest run failed and needs attention (for Results tab) */
  hasFailure?: boolean
}

export type QuickIngestSessionLifecycle =
  | "draft"
  | "processing"
  | "completed"
  | "partial_failure"
  | "cancelled"
  | "interrupted"

export type PersistedQuickIngestTracking = {
  mode: "webui-direct" | "extension-runtime" | "unknown"
  sessionId?: string
  batchId?: string
  batchIds?: string[]
  jobIds?: number[]
  submittedItemIds?: string[]
  /** @deprecated use submittedItemIds */
  itemIds?: string[]
  jobIdToItemId?: Record<string, string>
  startedAt?: number
}

export type ReattachedQuickIngestJob = {
  jobId: number
  status: string
  result?: unknown
  error?: string
  sourceItemId?: string
}

export type ReattachedQuickIngestSnapshot = {
  lifecycle: QuickIngestSessionLifecycle
  jobs: ReattachedQuickIngestJob[]
  errorMessage?: string | null
}

// ---------------------------------------------------------------------------
// Wizard types (ingest wizard redesign)
// ---------------------------------------------------------------------------

/**
 * Step number in the ingest wizard flow.
 * 1=Add, 2=Configure, 3=Review, 4=Processing, 5=Results
 */
export type WizardStep = 1 | 2 | 3 | 4 | 5

/**
 * Detected media type for a queued item.
 */
export type DetectedMediaType =
  | "audio"
  | "video"
  | "document"
  | "pdf"
  | "ebook"
  | "image"
  | "web"
  | "unknown"

/**
 * Validation state for a queued item.
 */
export type QueueItemValidation = {
  valid: boolean
  errors?: string[]
  warnings?: string[]
}

/**
 * An item in the wizard's ingest queue (files + URLs with detected types).
 */
export type WizardQueueItem = {
  /** Unique identifier for this queue item. */
  id: string
  /** Persisted queue item kind for refresh restore. */
  kind?: "url" | "file"
  /** Original file name (for file uploads). */
  fileName?: string
  /** URL string (for URL-based items). */
  url?: string
  /** The File object if this is a local file upload. */
  file?: File
  /** Persisted file stub metadata used to signal reattach-after-refresh. */
  fileStub?: {
    key?: string
    instanceId?: string
    lastModified?: number
  }
  /** Detected media type based on extension/MIME. */
  detectedType: DetectedMediaType
  /** Icon identifier (lucide icon name) for the detected type. */
  icon: string
  /** File size in bytes (0 for URLs until resolved). */
  fileSize: number
  /** MIME type if known. */
  mimeType?: string
  /** Validation state for this item. */
  validation: QueueItemValidation
}

/**
 * Processing status for a single item during the wizard's processing step.
 */
export type ItemProgressStatus =
  | "queued"
  | "uploading"
  | "processing"
  | "analyzing"
  | "storing"
  | "complete"
  | "failed"
  | "cancelled"

/**
 * Per-item progress tracking during processing.
 */
export type ItemProgress = {
  /** ID matching the corresponding WizardQueueItem. */
  id: string
  /** Current processing status of this item. */
  status: ItemProgressStatus
  /** Progress percentage (0-100). */
  progressPercent: number
  /** Human-readable label for the current processing stage. */
  currentStage: string
  /** Estimated seconds remaining for this item. */
  estimatedRemaining: number
  /** Error message if status is 'failed'. */
  error?: string
}

/**
 * Overall processing status for the wizard.
 */
export type ProcessingStatus =
  | "idle"
  | "running"
  | "complete"
  | "cancelled"
  | "error"

/**
 * Error classification for result items.
 */
export type ErrorClassification =
  | "network"
  | "auth"
  | "validation"
  | "server"
  | "timeout"
  | "unknown"

/**
 * Extended result item with error classification for the wizard.
 */
export type WizardResultItem = ResultItem & {
  /** Classification of the error, if status is "error". */
  errorClassification?: ErrorClassification
  /** Duration of processing in milliseconds. */
  durationMs?: number
  /** Media ID returned from the server. */
  mediaId?: string | number | null
  /** Title extracted or assigned during processing. */
  title?: string | null
  /** Non-error informational message (e.g., "Already exists in library"). */
  message?: string
}

/**
 * Aggregate processing state for the wizard.
 */
export type WizardProcessingState = {
  /** Overall status. */
  status: ProcessingStatus
  /** Per-item progress entries. */
  perItemProgress: ItemProgress[]
  /** Elapsed time in seconds since processing started. */
  elapsed: number
  /** Estimated total seconds remaining. */
  estimatedRemaining: number
}
