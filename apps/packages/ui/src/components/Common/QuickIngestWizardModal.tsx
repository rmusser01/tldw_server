import React, { useCallback, useEffect, useMemo, useRef } from "react"
import { Modal, Button } from "antd"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { XCircle } from "lucide-react"
import { browser } from "wxt/browser"
import {
  IngestWizardProvider,
  useIngestWizard,
  type IngestWizardState,
} from "./QuickIngest/IngestWizardContext"
import { IngestWizardStepper } from "./QuickIngest/IngestWizardStepper"
import { AddContentStep } from "./QuickIngest/AddContentStep"
import { WizardConfigureStep } from "./QuickIngest/WizardConfigureStep"
import { ReviewStep } from "./QuickIngest/ReviewStep"
import { ProcessingStep } from "./QuickIngest/ProcessingStep"
import { WizardResultsStep } from "./QuickIngest/WizardResultsStep"
import { FloatingProgressWidget } from "./QuickIngest/FloatingProgressWidget"
import {
  cancelQuickIngestSession,
  startQuickIngestSession,
  submitQuickIngestBatch,
} from "@/services/tldw/quick-ingest-batch"
import { reattachQuickIngestSession } from "@/services/tldw/quick-ingest-session-reattach"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  completedIngestJobIndicatesFailure,
  completedIngestJobIndicatesSkipped,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
} from "@/services/tldw/ingest-job-results"
import { DOCUMENT_WORKSPACE_PATH } from "@/routes/route-paths"
import {
  type PersistedWizardQueueItem,
  type QuickIngestSessionLifecycle,
  type QuickIngestSessionRecord,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"
import { useQuickIngestStore } from "@/store/quick-ingest"
import type {
  DetectedMediaType,
  ItemProgress,
  ItemProgressStatus,
  PersistedQuickIngestTracking,
  TypeDefaults,
  WizardQueueItem,
  WizardResultItem,
} from "./QuickIngest/types"
import {
  DUPLICATE_SKIP_MESSAGE,
  isDbMessageDuplicate,
} from "./QuickIngest/constants"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type QuickIngestWizardModalProps = {
  open: boolean
  onClose: () => void
  /** When true, automatically skip to processing on mount (compat with old modal). */
  autoProcessQueued?: boolean
}

type QuickIngestEntryType = "auto" | "html" | "pdf" | "document" | "audio" | "video"

type QuickIngestRequestPayload = {
  entries: Array<{
    id: string
    url: string
    type: QuickIngestEntryType
    defaults?: TypeDefaults
  }>
  files: Array<{
    id: string
    name: string
    type?: string
    data: number[]
    defaults?: TypeDefaults
  }>
  storeRemote: boolean
  processOnly: boolean
  common: {
    perform_analysis: boolean
    perform_chunking: boolean
    overwrite_existing: boolean
  }
  advancedValues: Record<string, unknown>
  fileDefaults: TypeDefaults
  __quickIngestSessionId?: string
}

type QuickIngestRuntimeMessage = {
  type: string
  payload?: {
    sessionId?: string
    result?: Partial<WizardResultItem>
    results?: Array<Partial<WizardResultItem>>
    error?: string
    reason?: string
  }
}

const RESULT_SUCCESS_STATUS_TOKENS = [
  "ok",
  "success",
  "completed",
  "complete",
  "done",
  "ingested",
  "processed",
  "ready",
]

const RESULT_CANCELLED_STATUS_TOKENS = ["cancelled", "canceled"]
const FILE_REATTACH_WARNING = "Reattach this file after refresh to process it."
const PERSISTED_REATTACH_POLL_INTERVAL_MS = 1_500

const mapDetectedTypeToEntryType = (
  detectedType: DetectedMediaType
): QuickIngestEntryType => {
  switch (detectedType) {
    case "audio":
    case "video":
    case "pdf":
    case "document":
      return detectedType
    case "web":
      return "html"
    default:
      return "auto"
  }
}

const buildDefaultsForQueueItem = (
  item: WizardQueueItem,
  typeDefaults: TypeDefaults
): TypeDefaults | undefined => {
  switch (item.detectedType) {
    case "audio":
      return typeDefaults.audio ? { audio: typeDefaults.audio } : undefined
    case "video":
      return {
        ...(typeDefaults.audio ? { audio: typeDefaults.audio } : {}),
        ...(typeDefaults.video ? { video: typeDefaults.video } : {}),
      }
    case "document":
    case "pdf":
    case "ebook":
    case "image":
      return typeDefaults.document ? { document: typeDefaults.document } : undefined
    default:
      return undefined
  }
}

const normalizeResultStatus = (status: unknown): "ok" | "error" => {
  const normalized = String(status || "").trim().toLowerCase()
  if (RESULT_SUCCESS_STATUS_TOKENS.includes(normalized)) return "ok"
  return "error"
}

const isCancelledError = (value: unknown) =>
  RESULT_CANCELLED_STATUS_TOKENS.some((token) =>
    String(value || "").trim().toLowerCase().includes(token)
  )

const normalizeWizardResult = (
  item: Partial<WizardResultItem> | null | undefined
): WizardResultItem | null => {
  if (!item?.id) return null
  const id = String(item.id).trim()
  if (!id) return null
  const status = normalizeResultStatus(item.status)
  const payloadFailure =
    status === "ok" && completedIngestJobIndicatesFailure(item.data)
  const derivedStatus = payloadFailure ? "error" : status
  const error =
    typeof item.error === "string"
      ? item.error
      : payloadFailure
        ? extractCompletedIngestJobError(item.data)
        : undefined
  const dataRecord = item.data != null && typeof item.data === "object"
    ? item.data as Record<string, unknown>
    : undefined
  const isDuplicate =
    derivedStatus === "ok" &&
    !item.outcome &&
    (
      isDbMessageDuplicate(dataRecord) ||
      completedIngestJobIndicatesSkipped(item.data)
    )
  return {
    id,
    status: derivedStatus,
    outcome:
      isDuplicate
        ? "skipped"
        : derivedStatus === "ok"
          ? item.outcome || "processed"
          : isCancelledError(error)
            ? "cancelled"
            : item.outcome || "failed",
    url: item.url,
    fileName: item.fileName,
    type: String(item.type || "item"),
    data: item.data,
    error,
    title: item.title,
    durationMs: item.durationMs,
    mediaId:
      item.mediaId ??
      extractCompletedIngestJobMediaId(item.data),
    message: isDuplicate
      ? DUPLICATE_SKIP_MESSAGE
      : typeof item.message === "string" ? item.message : undefined,
  }
}

const mergeWizardResults = (
  existing: WizardResultItem[],
  incoming: WizardResultItem[]
): WizardResultItem[] => {
  const merged = new Map<string, WizardResultItem>()
  for (const item of existing) {
    merged.set(item.id, item)
  }
  for (const item of incoming) {
    const previous = merged.get(item.id)
    merged.set(item.id, previous ? { ...previous, ...item } : item)
  }
  return Array.from(merged.values())
}

const buildTerminalProgress = (
  previous: ItemProgress,
  result: WizardResultItem
): ItemProgress => {
  const cancelled = result.outcome === "cancelled" || isCancelledError(result.error)
  const nextStatus: ItemProgressStatus =
    result.status === "ok" ? "complete" : cancelled ? "cancelled" : "failed"
  return {
    ...previous,
    status: nextStatus,
    progressPercent: 100,
    currentStage:
      nextStatus === "complete"
        ? "Complete"
        : nextStatus === "cancelled"
          ? "Cancelled"
          : result.error || "Failed",
    estimatedRemaining: 0,
    error: nextStatus === "failed" ? result.error : undefined,
  }
}

const buildFailureResults = (
  items: WizardQueueItem[],
  message: string,
  outcome: "failed" | "cancelled"
): WizardResultItem[] =>
  items.map((item) => ({
    id: item.id,
    status: "error",
    outcome,
    url: item.url,
    fileName: item.fileName,
    type: mapDetectedTypeToEntryType(item.detectedType),
    error: message,
  }))

const buildQueueFileKey = (item: WizardQueueItem): string | undefined => {
  if (item.fileStub?.key) return item.fileStub.key
  if (!item.file) return undefined
  const lastModified = Number.isFinite(item.file.lastModified) ? item.file.lastModified : 0
  return `${item.file.name}::${item.file.size}::${lastModified}`
}

const buildPersistedQueueItems = (
  items: WizardQueueItem[]
): PersistedWizardQueueItem[] =>
  items.map((item) => ({
    id: item.id,
    kind: item.kind || (item.url ? "url" : "file"),
    fileName: item.fileName || item.file?.name,
    name: item.fileName || item.file?.name,
    key: buildQueueFileKey(item),
    size: item.file ? item.file.size : item.fileSize,
    type: item.file?.type || item.mimeType,
    lastModified:
      item.file?.lastModified ?? item.fileStub?.lastModified ?? undefined,
    url: item.url,
    detectedType: item.detectedType,
    icon: item.icon,
    fileSize: item.file?.size ?? item.fileSize,
    mimeType: item.file?.type || item.mimeType,
    validation: item.validation,
    fileStub:
      item.file || item.fileStub
        ? {
            key: buildQueueFileKey(item),
            instanceId: item.fileStub?.instanceId,
            lastModified:
              item.file?.lastModified ?? item.fileStub?.lastModified ?? undefined,
          }
        : undefined,
  }))

const normalizeTrackedItemIds = (
  tracking?: Pick<PersistedQuickIngestTracking, "submittedItemIds" | "itemIds">
): string[] =>
  Array.from(
    new Set(
      [
        ...(Array.isArray(tracking?.submittedItemIds)
          ? tracking.submittedItemIds
          : []),
        ...(Array.isArray(tracking?.itemIds) ? tracking.itemIds : []),
      ]
        .map((itemId) => String(itemId || "").trim())
        .filter(Boolean)
    )
  )

const normalizeTrackedJobIds = (
  tracking?: Pick<PersistedQuickIngestTracking, "jobIds">
): number[] =>
  Array.from(
    new Set(
      (Array.isArray(tracking?.jobIds) ? tracking.jobIds : [])
        .map((jobId) => Number(jobId))
        .filter((jobId) => Number.isFinite(jobId) && jobId > 0)
        .map((jobId) => Math.trunc(jobId))
    )
  )

const resolveTrackedQueueItems = (
  items: WizardQueueItem[],
  tracking?: PersistedQuickIngestTracking
): WizardQueueItem[] => {
  const trackedItemIds = normalizeTrackedItemIds(tracking)
  if (trackedItemIds.length === 0) {
    const trackedJobIds = normalizeTrackedJobIds(tracking)
    if (trackedJobIds.length > 0) {
      return items.slice(0, trackedJobIds.length)
    }
    return items
  }
  const itemsById = new Map(items.map((item) => [item.id, item] as const))
  const trackedItems = trackedItemIds
    .map((itemId) => itemsById.get(itemId))
    .filter((item): item is WizardQueueItem => Boolean(item))
  return trackedItems.length > 0 ? trackedItems : items
}

const resolveQueueItemForReattachedJob = (
  items: WizardQueueItem[],
  tracking: PersistedQuickIngestTracking | undefined,
  sourceItemId: string | undefined,
  jobId: number,
  index: number
): WizardQueueItem | undefined => {
  const mappedItemId = String(sourceItemId || "").trim() || tracking?.jobIdToItemId?.[String(jobId)]
  if (mappedItemId) {
    return items.find((item) => item.id === mappedItemId)
  }
  return resolveTrackedQueueItems(items, tracking)[index]
}

const resolveTrackingBatchIds = (
  tracking?: PersistedQuickIngestTracking
): string[] =>
  Array.from(
    new Set(
      [tracking?.batchId, ...(tracking?.batchIds || [])]
        .map((batchId) => String(batchId || "").trim())
        .filter(Boolean)
    )
  )

const buildPersistedReattachSignature = (
  tracking?: PersistedQuickIngestTracking
): string => {
  if (!tracking) return ""

  const mode = String(tracking.mode || "unknown").trim() || "unknown"
  const sessionId = String(tracking.sessionId || "").trim()
  const batchIds = resolveTrackingBatchIds(tracking)
  const jobIds = normalizeTrackedJobIds(tracking)
  const itemIds = normalizeTrackedItemIds(tracking)
  const jobIdToItemId = Object.entries(tracking.jobIdToItemId ?? {})
    .map(([jobId, itemId]) => `${jobId}:${String(itemId || "").trim()}`)
    .sort()

  return [
    mode,
    sessionId,
    batchIds.join(","),
    jobIds.join(","),
    itemIds.join(","),
    jobIdToItemId.join(","),
  ].join("|")
}

const hydrateQueueItems = (
  queueItems: QuickIngestSessionRecord["queueItems"]
): WizardQueueItem[] =>
  queueItems.map((item) => {
    const isFileItem = item.kind === "file" || (!item.url && Boolean(item.fileName || item.name))
    if (!isFileItem) {
      return {
        id: item.id,
        kind: "url",
        url: item.url,
        detectedType: item.detectedType,
        icon: item.icon,
        fileSize: item.fileSize,
        mimeType: item.mimeType,
        validation: item.validation,
      }
    }

    const warnings = Array.from(
      new Set([...(item.validation.warnings ?? []), FILE_REATTACH_WARNING])
    )

    return {
      id: item.id,
      kind: "file",
      fileName: item.fileName || item.name,
      detectedType: item.detectedType,
      icon: item.icon,
      fileSize: item.fileSize,
      mimeType: item.mimeType || item.type,
      validation: {
        ...item.validation,
        valid: false,
        warnings,
      },
      fileStub: item.fileStub || {
        key: item.key,
        lastModified: item.lastModified,
      },
    }
  })

const deriveLifecycleFromWizardState = (
  state: IngestWizardState,
  existingLifecycle?: QuickIngestSessionLifecycle
): QuickIngestSessionLifecycle => {
  if (state.currentStep < 4 && state.processingState.status === "idle") {
    return "draft"
  }

  if (state.processingState.status === "running") {
    return "processing"
  }

  if (state.processingState.status === "cancelled") {
    return "cancelled"
  }

  if (state.processingState.status === "error") {
    if (existingLifecycle === "interrupted") {
      return "interrupted"
    }
    return "partial_failure"
  }

  if (state.processingState.status === "complete" || state.currentStep === 5) {
    const hasFailures = state.results.some(
      (item) => item.status === "error" || item.outcome === "failed"
    )
    const allCancelled =
      state.results.length > 0 &&
      state.results.every((item) => item.outcome === "cancelled")
    if (allCancelled) return "cancelled"
    return hasFailures ? "partial_failure" : "completed"
  }

  return existingLifecycle || "draft"
}

const buildResultSummaryFromState = (
  state: IngestWizardState,
  lifecycle: QuickIngestSessionLifecycle,
  existingSession: QuickIngestSessionRecord
): QuickIngestSessionRecord["resultSummary"] => {
  const successes = state.results.filter((item) => item.status === "ok")
  const failures = state.results.filter(
    (item) => item.status === "error" && item.outcome !== "cancelled"
  )
  const cancelled = state.results.filter((item) => item.outcome === "cancelled")
  const firstSuccess = successes[0]

  return {
    ...existingSession.resultSummary,
    status:
      lifecycle === "completed"
        ? "success"
        : lifecycle === "cancelled"
          ? "cancelled"
          : lifecycle === "partial_failure" || lifecycle === "interrupted"
            ? "error"
            : existingSession.resultSummary.status,
    attemptedAt:
      existingSession.resultSummary.attemptedAt ??
      existingSession.createdAt ??
      Date.now(),
    completedAt:
      lifecycle === "completed" ||
      lifecycle === "partial_failure" ||
      lifecycle === "cancelled" ||
      lifecycle === "interrupted"
        ? Date.now()
        : existingSession.resultSummary.completedAt,
    totalCount: state.results.length || state.queueItems.length,
    successCount: successes.length,
    failedCount: failures.length,
    cancelledCount: cancelled.length,
    firstMediaId:
      firstSuccess?.mediaId === null || typeof firstSuccess?.mediaId === "undefined"
        ? existingSession.resultSummary.firstMediaId
        : String(firstSuccess.mediaId),
    primarySourceLabel:
      firstSuccess?.title ||
      firstSuccess?.fileName ||
      firstSuccess?.url ||
      existingSession.resultSummary.primarySourceLabel,
    errorMessage:
      failures[0]?.error ||
      cancelled[0]?.error ||
      existingSession.errorMessage ||
      null,
  }
}

const buildInitialWizardState = (
  session: QuickIngestSessionRecord
): IngestWizardState => ({
  currentStep: session.currentStep,
  highestStep: Math.max(session.currentStep, 1) as IngestWizardState["highestStep"],
  queueItems: hydrateQueueItems(session.queueItems),
  selectedPreset: session.selectedPreset,
  customBasePreset: session.customBasePreset,
  presetConfig: session.presetConfig,
  customOptions: session.customOptions,
  processingState: session.processingState,
  results: session.results,
  isMinimized:
    session.visibility === "hidden" && session.lifecycle === "processing",
})

const buildSessionPatchFromWizardState = (
  state: IngestWizardState,
  session: QuickIngestSessionRecord
): Partial<QuickIngestSessionRecord> => {
  const lifecycle = deriveLifecycleFromWizardState(state, session.lifecycle)
  const queueItems = buildPersistedQueueItems(state.queueItems)
  return {
    currentStep: state.currentStep,
    queueItems,
    selectedPreset: state.selectedPreset,
    customBasePreset: state.customBasePreset,
    presetConfig: state.presetConfig,
    customOptions: state.customOptions,
    processingState: state.processingState,
    results: state.results,
    badge: {
      queueCount:
        lifecycle === "draft"
          ? queueItems.filter((item) => item.validation.valid).length
          : 0,
      hasRecentFailure:
        lifecycle === "partial_failure" || lifecycle === "interrupted",
    },
    lifecycle,
    completedAt:
      lifecycle === "completed" ||
      lifecycle === "partial_failure" ||
      lifecycle === "cancelled" ||
      lifecycle === "interrupted"
        ? Date.now()
        : null,
    errorMessage:
      lifecycle === "partial_failure" || lifecycle === "interrupted"
        ? state.results.find((item) => item.status === "error")?.error ||
          session.errorMessage ||
          null
        : lifecycle === "cancelled"
          ? state.results.find((item) => item.outcome === "cancelled")?.error || null
          : null,
    resultSummary: buildResultSummaryFromState(state, lifecycle, session),
  }
}

const mapReattachedJobStatusToProgress = (
  status: string,
  result?: unknown
): ItemProgressStatus => {
  switch (status) {
    case "pending":
    case "queued":
      return "queued"
    case "uploading":
      return "uploading"
    case "running":
    case "processing":
      return "processing"
    case "analyzing":
      return "analyzing"
    case "storing":
      return "storing"
    case "completed":
      if (completedIngestJobIndicatesFailure(result)) {
        return "failed"
      }
      return "complete"
    case "cancelled":
      return "cancelled"
    default:
      return "failed"
  }
}

const buildResultsFromReattachedJobs = (
  items: WizardQueueItem[],
  jobs: Array<{
    jobId: number
    status: string
    result?: any
    error?: string
    sourceItemId?: string
  }>,
  tracking?: PersistedQuickIngestTracking
): WizardResultItem[] =>
  jobs.map((job, index) => {
    const item = resolveQueueItemForReattachedJob(
      items,
      tracking,
      job.sourceItemId,
      job.jobId,
      index
    )
    const jobStatus = String(job.status || "").trim().toLowerCase()
    const logicalFailure =
      jobStatus === "completed" && completedIngestJobIndicatesFailure(job.result)
    const resultStatus =
      jobStatus === "completed" && !logicalFailure ? "ok" : "error"
    const isDuplicate = resultStatus === "ok" && completedIngestJobIndicatesSkipped(job.result)
    return {
      id: item?.id || `reattached-${job.jobId}`,
      status: resultStatus,
      outcome:
        isDuplicate
          ? "skipped" as const
          : resultStatus === "ok"
            ? "processed"
            : jobStatus === "cancelled"
              ? "cancelled"
              : "failed",
      url: item?.url,
      fileName: item?.fileName,
      type: mapDetectedTypeToEntryType(item?.detectedType || "unknown"),
      error:
        resultStatus === "ok"
          ? undefined
          : job.error ||
            extractCompletedIngestJobError(job.result) ||
            `Quick ingest ${jobStatus || "failed"}.`,
      mediaId: extractCompletedIngestJobMediaId(job.result),
      title: job.result?.title ?? null,
      data: job.result,
      message: isDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
    }
  })

const buildProgressFromReattachedJobs = (
  items: WizardQueueItem[],
  jobs: Array<{ jobId: number; status: string; error?: string; sourceItemId?: string }>,
  tracking?: PersistedQuickIngestTracking
): ItemProgress[] =>
  jobs.map((job, index) => {
    const item = resolveQueueItemForReattachedJob(
      items,
      tracking,
      job.sourceItemId,
      job.jobId,
      index
    )
    const status = mapReattachedJobStatusToProgress(job.status, job.result)
    const progressPercent =
      status === "complete" || status === "failed" || status === "cancelled"
        ? 100
        : status === "queued"
          ? 0
          : 50
    return {
      id: item?.id || `reattached-${job.jobId}`,
      status,
      progressPercent,
      currentStage:
        status === "failed"
          ? job.error ||
            extractCompletedIngestJobError(job.result) ||
            "Failed"
          : status === "complete"
            ? "Complete"
            : status === "cancelled"
              ? "Cancelled"
              : String(job.status || "Processing"),
      estimatedRemaining: 0,
      error: status === "failed" ? job.error : undefined,
    }
  })

const buildQuickIngestPayload = async (
  items: WizardQueueItem[],
  options: QuickIngestRequestPayload["common"] & {
    storeRemote: boolean
    reviewBeforeStorage: boolean
    advancedValues?: Record<string, unknown>
    typeDefaults: TypeDefaults
  }
): Promise<QuickIngestRequestPayload> => {
  const validItems = items.filter((item) => item.validation.valid)
  const entries = validItems
    .filter((item): item is WizardQueueItem & { url: string } => Boolean(item.url))
    .map((item) => ({
      id: item.id,
      url: item.url,
      type: mapDetectedTypeToEntryType(item.detectedType),
      defaults: buildDefaultsForQueueItem(item, options.typeDefaults),
    }))

  const files = await Promise.all(
    validItems
      .filter((item): item is WizardQueueItem & { file: File } => Boolean(item.file))
      .map(async (item) => ({
        id: item.id,
        name: item.file.name,
        type: item.file.type || undefined,
        data: Array.from(new Uint8Array(await item.file.arrayBuffer())),
        defaults: buildDefaultsForQueueItem(item, options.typeDefaults),
      }))
  )

  return {
    entries,
    files,
    storeRemote: options.storeRemote,
    processOnly: options.reviewBeforeStorage || !options.storeRemote,
    common: {
      perform_analysis: options.perform_analysis,
      perform_chunking: options.perform_chunking,
      overwrite_existing: options.overwrite_existing,
    },
    advancedValues: options.advancedValues ?? {},
    fileDefaults: options.typeDefaults,
  }
}

// ---------------------------------------------------------------------------
// Inner modal content (must be inside IngestWizardProvider)
// ---------------------------------------------------------------------------

type WizardModalContentProps = {
  open: boolean
  onClose: () => void
  autoProcessQueued?: boolean
  session: QuickIngestSessionRecord
  markProcessingTracking: (tracking: PersistedQuickIngestTracking) => void
  markInterrupted: (reason?: string) => void
  shouldAttemptPersistedReattach: boolean
}

const WizardModalContent: React.FC<WizardModalContentProps> = ({
  open,
  onClose,
  autoProcessQueued = false,
  session,
  markProcessingTracking,
  markInterrupted,
  shouldAttemptPersistedReattach,
}) => {
  const { t } = useTranslation(["option"])
  const {
    state,
    minimize,
    restore,
    cancelProcessing,
    skipToProcessing,
    updateItemProgress,
    updateProcessingState,
    setResults,
    goNext,
  } = useIngestWizard()
  const { currentStep, queueItems, processingState, presetConfig, results } = state
  const activeSessionIdRef = useRef<string | null>(null)
  const resultsRef = useRef(results)
  const hasStartedRunRef = useRef(false)
  const runStartedAtRef = useRef<number | null>(null)
  const cancelledSessionIdsRef = useRef<Set<string>>(new Set())
  const validQueueItems = useMemo(
    () => queueItems.filter((item) => item.validation.valid),
    [queueItems]
  )
  const trackedQueueItems = useMemo(
    () => resolveTrackedQueueItems(queueItems, session.tracking),
    [queueItems, session.tracking]
  )
  const initialTrackedQueueItemsRef = useRef(trackedQueueItems)
  const initialCurrentStepRef = useRef(currentStep)
  const initialElapsedRef = useRef(state.processingState.elapsed)
  const persistedTrackingRef = useRef(session.tracking)
  const persistedReattachTimerRef = useRef<number | null>(null)
  const activeReattachSignatureRef = useRef("")
  const persistedReattachSignature = useMemo(
    () =>
      shouldAttemptPersistedReattach
        ? buildPersistedReattachSignature(session.tracking)
        : "",
    [session.tracking, shouldAttemptPersistedReattach]
  )

  useEffect(() => {
    resultsRef.current = results
  }, [results])

  useEffect(() => {
    initialTrackedQueueItemsRef.current = trackedQueueItems
  }, [trackedQueueItems])

  useEffect(() => {
    persistedTrackingRef.current = session.tracking
    const sessionId = String(session.tracking?.sessionId || "").trim()
    if (sessionId) {
      activeSessionIdRef.current = sessionId
    }
    const startedAt = session.tracking?.startedAt
    if (
      typeof startedAt === "number" &&
      Number.isFinite(startedAt) &&
      !runStartedAtRef.current
    ) {
      runStartedAtRef.current = startedAt
    }
  }, [session.tracking])

  useEffect(() => {
    if (!shouldAttemptPersistedReattach) {
      activeReattachSignatureRef.current = ""
    }
  }, [shouldAttemptPersistedReattach])

  // Track recently ingested documents for the DocumentPickerModal
  const addRecentlyIngestedDocs = useQuickIngestStore(s => s.addRecentlyIngestedDocs)
  const recordedIngestRef = useRef(false)

  useEffect(() => {
    if (state.currentStep !== 5) { recordedIngestRef.current = false; return }
    if (recordedIngestRef.current) return
    recordedIngestRef.current = true
    const newDocs = state.results
      .filter(
        (item) =>
          item.status === "ok" &&
          item.mediaId != null &&
          ["pdf", "ebook", "document"].includes(item.type)
      )
      .map((item) => {
        const id = Number(item.mediaId)
        return Number.isFinite(id) && id > 0
          ? { id, type: item.type, title: item.title || undefined }
          : null
      })
      .filter((d): d is NonNullable<typeof d> => d != null)
    if (newDocs.length > 0) {
      addRecentlyIngestedDocs(newDocs)
    }
  }, [state.currentStep, state.results, addRecentlyIngestedDocs])

  useEffect(() => {
    if (!open || !state.isMinimized) return
    restore()
  }, [open, restore, state.isMinimized])

  // Auto-process on mount if autoProcessQueued is set and there are queued items
  const autoProcessedRef = useRef(false)
  useEffect(() => {
    if (autoProcessQueued && !autoProcessedRef.current && queueItems.length > 0) {
      autoProcessedRef.current = true
      skipToProcessing()
    }
  }, [autoProcessQueued, queueItems.length, skipToProcessing])

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t],
  )

  // Whether processing is actively running
  const isProcessingActive = processingState.status === "running"

  const syncElapsed = useCallback(() => {
    const startedAt = runStartedAtRef.current
    if (!startedAt) return
    updateProcessingState({
      elapsed: Math.max(0, Math.floor((Date.now() - startedAt) / 1000)),
    })
  }, [updateProcessingState])

  useEffect(() => {
    if (processingState.status !== "running") return
    syncElapsed()
    const timer = window.setInterval(syncElapsed, 1000)
    return () => window.clearInterval(timer)
  }, [processingState.status, syncElapsed])

  // ---------------------------------------------------------------------------
  // Simulated progress advancement
  //
  // During long-running server-side processing (transcription, indexing) the
  // UI receives no intermediate progress updates -- items stay stuck at 10%.
  // This effect gradually advances non-terminal items through visual stages
  // so the user sees continuous feedback while waiting.
  //
  // We use a ref to read the latest perItemProgress inside the interval
  // callback so the effect only depends on `processingState.status` (avoiding
  // a re-render feedback loop where updating progress re-triggers the effect).
  // ---------------------------------------------------------------------------
  const perItemProgressRef = useRef(processingState.perItemProgress)
  useEffect(() => {
    perItemProgressRef.current = processingState.perItemProgress
  }, [processingState.perItemProgress])

  const updateItemProgressRef = useRef(updateItemProgress)
  useEffect(() => {
    updateItemProgressRef.current = updateItemProgress
  }, [updateItemProgress])

  useEffect(() => {
    if (processingState.status !== "running") return

    const STAGE_ORDER: ItemProgressStatus[] = [
      "uploading",
      "processing",
      "analyzing",
      "storing",
    ]
    const TERMINAL = new Set<ItemProgressStatus>(["complete", "failed", "cancelled"])
    /** Advance interval in ms -- slow enough to feel realistic. */
    const TICK_MS = 3_000
    /**
     * Maximum simulated percent -- we cap at 90% so the bar never reaches
     * 100% before the real server response arrives.
     */
    const MAX_SIMULATED_PERCENT = 90
    /** Percent increment per tick. */
    const PERCENT_STEP = 5

    // TODO(i18n): extract STAGE_LABELS to i18n resources
    const STAGE_LABELS: Record<string, string> = {
      uploading: "Uploading",
      processing: "Processing your file... This may take a few minutes for large files.",
      analyzing: "Transcribing and indexing content",
      storing: "Storing results",
    }

    const timer = window.setInterval(() => {
      const items = perItemProgressRef.current
      for (const p of items) {
        if (TERMINAL.has(p.status)) continue

        const currentStageIdx = STAGE_ORDER.indexOf(p.status)
        if (currentStageIdx < 0) continue

        let nextPercent = p.progressPercent + PERCENT_STEP
        let nextStatus = p.status
        let nextStage = p.currentStage

        // Advance to the next visual stage at certain thresholds
        if (nextPercent >= 35 && currentStageIdx === 0) {
          nextStatus = "processing"
          nextStage = STAGE_LABELS.processing
        } else if (nextPercent >= 60 && currentStageIdx <= 1) {
          nextStatus = "analyzing"
          nextStage = STAGE_LABELS.analyzing
        } else if (nextPercent >= 80 && currentStageIdx <= 2) {
          nextStatus = "storing"
          nextStage = STAGE_LABELS.storing
        } else {
          nextStage = STAGE_LABELS[p.status] || p.currentStage
        }

        if (nextPercent > MAX_SIMULATED_PERCENT) {
          nextPercent = MAX_SIMULATED_PERCENT
        }

        // Only update if something actually changed
        if (
          nextPercent !== p.progressPercent ||
          nextStatus !== p.status ||
          nextStage !== p.currentStage
        ) {
          updateItemProgressRef.current({
            ...p,
            status: nextStatus,
            progressPercent: nextPercent,
            currentStage: nextStage,
          })
        }
      }
    }, TICK_MS)

    return () => window.clearInterval(timer)
  }, [processingState.status])

  useEffect(() => {
    const persistedTracking = persistedTrackingRef.current
    const reattachQueueItems = initialTrackedQueueItemsRef.current
    if (!persistedReattachSignature || !persistedTracking) return
    if (activeReattachSignatureRef.current === persistedReattachSignature) return
    activeReattachSignatureRef.current = persistedReattachSignature

    const startedAt = persistedTracking.startedAt
    if (typeof startedAt === "number" && Number.isFinite(startedAt)) {
      runStartedAtRef.current = startedAt
    }
    const sessionId = String(persistedTracking.sessionId || "").trim()
    if (sessionId) {
      activeSessionIdRef.current = sessionId
    }

    let cancelled = false
    const pollPersistedTracking = async () => {
      const currentTracking = persistedTrackingRef.current || persistedTracking
      const snapshot = await reattachQuickIngestSession(currentTracking)
      if (cancelled) return

      const latestTracking = persistedTrackingRef.current || currentTracking
      const perItemProgress = buildProgressFromReattachedJobs(
        reattachQueueItems,
        snapshot.jobs,
        latestTracking
      )
      const elapsed =
        typeof startedAt === "number" && Number.isFinite(startedAt)
          ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
          : initialElapsedRef.current

      if (snapshot.lifecycle === "processing") {
        updateProcessingState({
          status: "running",
          perItemProgress,
          elapsed,
          estimatedRemaining: 0,
        })
        persistedReattachTimerRef.current = window.setTimeout(() => {
          void pollPersistedTracking()
        }, PERSISTED_REATTACH_POLL_INTERVAL_MS)
        return
      }

      const reattachedResults =
        snapshot.jobs.length > 0
          ? buildResultsFromReattachedJobs(
              reattachQueueItems,
              snapshot.jobs,
              latestTracking
            )
          : buildFailureResults(
              reattachQueueItems,
              snapshot.errorMessage || "Quick ingest could not reconnect to live job status.",
              "failed"
            )

      resultsRef.current = reattachedResults
      setResults(reattachedResults)
      updateProcessingState({
        status:
          snapshot.lifecycle === "completed"
            ? "complete"
            : snapshot.lifecycle === "cancelled"
              ? "cancelled"
              : "error",
        perItemProgress,
        elapsed,
        estimatedRemaining: 0,
      })
      hasStartedRunRef.current = false
      activeSessionIdRef.current = null
      if (snapshot.lifecycle === "interrupted") {
        markInterrupted(
          snapshot.errorMessage || "Quick ingest could not reconnect to live job status."
        )
      }
      if (initialCurrentStepRef.current < 5) {
        goNext()
      }
    }

    void pollPersistedTracking()

    return () => {
      cancelled = true
      if (persistedReattachTimerRef.current != null) {
        window.clearTimeout(persistedReattachTimerRef.current)
        persistedReattachTimerRef.current = null
      }
    }
  }, [
    goNext,
    markInterrupted,
    persistedReattachSignature,
    setResults,
    updateProcessingState,
  ])

  const applyResults = useCallback(
    (incoming: WizardResultItem[]) => {
      const next = mergeWizardResults(resultsRef.current, incoming)
      resultsRef.current = next
      setResults(next)
      const progressMap = new Map(
        processingState.perItemProgress.map((item) => [item.id, item])
      )
      for (const result of incoming) {
        const previous = progressMap.get(result.id)
        if (!previous) continue
        updateItemProgress(buildTerminalProgress(previous, result))
      }
      return next
    },
    [processingState.perItemProgress, setResults, updateItemProgress]
  )

  const finalizeRun = useCallback(
    (
      nextStatus: "complete" | "cancelled" | "error",
      incomingResults: WizardResultItem[]
    ) => {
      syncElapsed()
      applyResults(incomingResults)
      updateProcessingState({
        status: nextStatus,
        estimatedRemaining: 0,
      })
      hasStartedRunRef.current = false
      activeSessionIdRef.current = null
      goNext()
    },
    [applyResults, goNext, syncElapsed, updateProcessingState]
  )

  const finalizeFailure = useCallback(
    (message: string, outcome: "failed" | "cancelled") => {
      const fallbackItems =
        trackedQueueItems.length > 0 ? trackedQueueItems : validQueueItems
      const existingResultIds = new Set(
        resultsRef.current
          .map((result) => String(result.id || "").trim())
          .filter(Boolean)
      )
      const unresolvedFallbackItems = fallbackItems.filter(
        (item) => !existingResultIds.has(item.id)
      )
      const fallbackResults = buildFailureResults(
        unresolvedFallbackItems,
        message,
        outcome
      )
      finalizeRun(outcome === "cancelled" ? "cancelled" : "error", fallbackResults)
    },
    [finalizeRun, trackedQueueItems, validQueueItems]
  )

  const markRunActive = useCallback(() => {
    runStartedAtRef.current = Date.now()
    for (const item of validQueueItems) {
      const initialStatus: ItemProgressStatus = item.file ? "uploading" : "processing"
      updateItemProgress({
        id: item.id,
        status: initialStatus,
        progressPercent: 10,
        currentStage: initialStatus === "uploading" ? "Uploading" : "Processing",
        estimatedRemaining: 0,
      })
    }
  }, [updateItemProgress, validQueueItems])

  const handleRuntimeMessage = useCallback(
    (message: QuickIngestRuntimeMessage) => {
      if (!message || typeof message.type !== "string") return
      const sessionId = String(message.payload?.sessionId || "").trim()
      if (!sessionId || sessionId !== String(activeSessionIdRef.current || "").trim()) {
        return
      }
      if (
        cancelledSessionIdsRef.current.has(sessionId) &&
        message.type !== "tldw:quick-ingest/progress"
      ) {
        return
      }

      if (message.type === "tldw:quick-ingest/progress") {
        const result = normalizeWizardResult(message.payload?.result)
        if (result) {
          applyResults([result])
        }
        return
      }

      if (message.type === "tldw:quick-ingest/completed") {
        const normalizedResults = (message.payload?.results || [])
          .map((item) => normalizeWizardResult(item))
          .filter((item): item is WizardResultItem => Boolean(item))
        if (normalizedResults.length === 0) {
          finalizeFailure("Ingest request finished without item results.", "failed")
          return
        }
        finalizeRun("complete", normalizedResults)
        return
      }

      if (message.type === "tldw:quick-ingest/failed") {
        finalizeFailure(
          String(message.payload?.error || "Quick ingest failed."),
          "failed"
        )
        return
      }

      if (message.type === "tldw:quick-ingest/cancelled") {
        finalizeFailure(
          String(message.payload?.reason || "Cancelled by user."),
          "cancelled"
        )
      }
    },
    [applyResults, finalizeFailure, finalizeRun]
  )

  useEffect(() => {
    const listener = (message: QuickIngestRuntimeMessage) => {
      handleRuntimeMessage(message)
    }
    try {
      if (browser?.runtime?.onMessage?.addListener) {
        browser.runtime.onMessage.addListener(listener)
      }
    } catch {
      return
    }

    return () => {
      try {
        if (browser?.runtime?.onMessage?.removeListener) {
          browser.runtime.onMessage.removeListener(listener)
        }
      } catch {
        // Ignore cleanup failures in non-extension runtimes.
      }
    }
  }, [handleRuntimeMessage])

  const startRun = useCallback(async () => {
    if (hasStartedRunRef.current || validQueueItems.length === 0) return
    hasStartedRunRef.current = true
    markRunActive()

    try {
      try {
        await tldwClient.initialize()
      } catch {
        // Best effort; background proxy handles auth for direct runtimes.
      }

      const requestPayload = await buildQuickIngestPayload(validQueueItems, {
        ...presetConfig.common,
        storeRemote: presetConfig.storeRemote,
        reviewBeforeStorage: presetConfig.reviewBeforeStorage,
        advancedValues: presetConfig.advancedValues,
        typeDefaults: presetConfig.typeDefaults,
      })

      const startAck = await startQuickIngestSession(requestPayload)
      if (!startAck?.ok || !startAck?.sessionId) {
        finalizeFailure(
          startAck?.error ||
            "Quick ingest failed to start. Check tldw server settings and try again.",
          "failed"
        )
        return
      }

      const sessionId = String(startAck.sessionId).trim()
      activeSessionIdRef.current = sessionId
      cancelledSessionIdsRef.current.delete(sessionId)
      markProcessingTracking({
        mode: sessionId.startsWith("qi-direct-") ? "webui-direct" : "extension-runtime",
        sessionId,
        startedAt: runStartedAtRef.current || Date.now(),
      })

      if (!sessionId.startsWith("qi-direct-")) {
        return
      }

      const response = await submitQuickIngestBatch({
        ...requestPayload,
        __quickIngestSessionId: sessionId,
        onTrackingMetadata: (tracking) => {
          markProcessingTracking({
            ...tracking,
            sessionId,
            mode: "webui-direct",
            startedAt: tracking.startedAt || runStartedAtRef.current || Date.now(),
          })
        },
      })

      if (
        cancelledSessionIdsRef.current.has(sessionId) ||
        sessionId !== String(activeSessionIdRef.current || "").trim()
      ) {
        return
      }

      if (!response?.ok) {
        finalizeFailure(
          response?.error ||
            "Quick ingest failed. Check tldw server settings and try again.",
          "failed"
        )
        return
      }

      const normalizedResults = (response.results || [])
        .map((item) => normalizeWizardResult(item))
        .filter((item): item is WizardResultItem => Boolean(item))

      if (normalizedResults.length === 0) {
        finalizeFailure("Ingest request finished without item results.", "failed")
        return
      }

      finalizeRun("complete", normalizedResults)
    } catch (error) {
      finalizeFailure(
        error instanceof Error ? error.message : "Quick ingest failed.",
        "failed"
      )
    }
  }, [
    finalizeFailure,
    finalizeRun,
    markRunActive,
    presetConfig.advancedValues,
    presetConfig.common,
    presetConfig.reviewBeforeStorage,
    presetConfig.storeRemote,
    presetConfig.typeDefaults,
    markProcessingTracking,
    validQueueItems,
  ])

  useEffect(() => {
    if (currentStep !== 4 || processingState.status !== "running") return
    if (session.lifecycle === "processing" && session.tracking) {
      const canReattachDirectJobs =
        session.tracking.mode === "webui-direct" &&
        Boolean(session.tracking.jobIds?.length)
      if (session.tracking.mode !== "webui-direct" || canReattachDirectJobs) {
        return
      }
    }
    void startRun()
  }, [currentStep, processingState.status, session.lifecycle, session.tracking, startRun])

  useEffect(() => {
    if (processingState.status !== "cancelled") return
    const persistedTracking = persistedTrackingRef.current
    const sessionId = String(
      activeSessionIdRef.current || persistedTracking?.sessionId || ""
    ).trim()
    if (!sessionId || cancelledSessionIdsRef.current.has(sessionId)) return
    cancelledSessionIdsRef.current.add(sessionId)
    void cancelQuickIngestSession({
      sessionId,
      batchIds: resolveTrackingBatchIds(persistedTracking),
      reason: "user_cancelled",
    }).catch(() => {
      // best effort cancellation
    })
    finalizeFailure("Cancelled by user.", "cancelled")
  }, [finalizeFailure, processingState.status])

  // Modal title with item count
  const modalTitle = useMemo(() => {
    const base = qi("wizard.title", "Quick Ingest")
    if (queueItems.length > 0 && currentStep <= 3) {
      return `${base} (${queueItems.length})`
    }
    return base
  }, [qi, queueItems.length, currentStep])

  // Close handler with confirmation when processing
  const handleCloseAttempt = useCallback(() => {
    if (isProcessingActive) {
      Modal.confirm({
        title: qi(
          "wizard.closeConfirm.title",
          "Processing is in progress",
        ),
        content: qi(
          "wizard.closeConfirm.content",
          "Would you like to minimize to background or cancel all items?",
        ),
        okText: qi("wizard.closeConfirm.minimize", "Minimize to Background"),
        okButtonProps: { type: "primary" },
        cancelText: qi("wizard.closeConfirm.stay", "Stay"),
        footer: (_, { OkBtn, CancelBtn }) => (
          <div className="flex items-center justify-end gap-2">
            <CancelBtn />
            <Button
              danger
              onClick={() => {
                Modal.destroyAll()
                cancelProcessing()
                onClose()
              }}
            >
              <XCircle className="mr-1 h-4 w-4" />
              {qi("wizard.closeConfirm.cancelAll", "Cancel All")}
            </Button>
            <OkBtn />
          </div>
        ),
        onOk: () => {
          minimize()
          onClose()
        },
        icon: null,
        maskClosable: true,
      })
    } else {
      onClose()
    }
  }, [isProcessingActive, qi, minimize, cancelProcessing, onClose])

  // Quick-process callback for AddContentStep (skip to processing with defaults)
  const handleQuickProcess = useCallback(() => {
    skipToProcessing()
  }, [skipToProcessing])

  // Navigation callbacks for WizardResultsStep CTAs
  const navigate = useNavigate()
  const mountedRef = useRef(true)
  useEffect(() => () => { mountedRef.current = false }, [])

  const handleSearchKnowledge = useCallback(() => {
    onClose()
    window.setTimeout(() => { if (mountedRef.current) navigate("/knowledge") }, 150)
  }, [navigate, onClose])

  const handleOpenWorkspace = useCallback(
    (item: WizardResultItem) => {
      onClose()
      const mediaId = item.mediaId
      if (mediaId != null) {
        window.setTimeout(
          () => { if (mountedRef.current) navigate(`${DOCUMENT_WORKSPACE_PATH}?open=${mediaId}`) },
          150
        )
      } else {
        window.setTimeout(() => { if (mountedRef.current) navigate(DOCUMENT_WORKSPACE_PATH) }, 150)
      }
    },
    [navigate, onClose]
  )

  // Render the current step
  const stepContent = useMemo(() => {
    switch (currentStep) {
      case 1:
        return <AddContentStep onQuickProcess={handleQuickProcess} />
      case 2:
        return (
          <WizardConfigureStep
            isStepVisible={open && !state.isMinimized && currentStep === 2}
          />
        )
      case 3:
        return <ReviewStep />
      case 4:
        return <ProcessingStep />
      case 5:
        return (
          <WizardResultsStep
            onClose={onClose}
            onSearchKnowledge={handleSearchKnowledge}
            onOpenWorkspace={handleOpenWorkspace}
          />
        )
      default:
        return null
    }
  }, [currentStep, handleQuickProcess, handleSearchKnowledge, handleOpenWorkspace, onClose, open, state.isMinimized])

  return (
    <>
      <Modal
        open={open && !state.isMinimized}
        onCancel={handleCloseAttempt}
        title={modalTitle}
        footer={null}
        width={800}
        className="quick-ingest-modal quick-ingest-wizard-modal"
        styles={{
          body: {
            padding: "0 16px 16px",
            maxHeight: "calc(100vh - 180px)",
            overflowY: "auto",
          },
        }}
      >
        {/* Stepper navigation */}
        <IngestWizardStepper />

        {/* Step content */}
        <div className="min-h-[300px]">{stepContent}</div>
      </Modal>

      {/* Floating progress widget (renders via portal when minimized) */}
      <FloatingProgressWidget />
    </>
  )
}

// ---------------------------------------------------------------------------
// Exported modal component
// ---------------------------------------------------------------------------

export const QuickIngestWizardModal: React.FC<QuickIngestWizardModalProps> = ({
  open,
  onClose,
  autoProcessQueued = false,
}) => {
  const {
    session,
    upsertSession,
    markProcessingTracking,
    markInterrupted,
    createDraftSession,
  } =
    useQuickIngestSessionStore((store) => ({
      session: store.session,
      upsertSession: store.upsertSession,
      markProcessingTracking: store.markProcessingTracking,
      markInterrupted: store.markInterrupted,
      createDraftSession: store.createDraftSession,
    }))

  const initialState = useMemo(
    () => (session ? buildInitialWizardState(session) : undefined),
    [session]
  )
  const sessionRef = useRef(session)

  useEffect(() => {
    sessionRef.current = session
  }, [session])

  useEffect(() => {
    if (!open || session) return
    createDraftSession()
  }, [createDraftSession, open, session])

  const persistWizardState = useCallback(
    (state: IngestWizardState) => {
      const currentSession = sessionRef.current
      if (!currentSession) return
      upsertSession(buildSessionPatchFromWizardState(state, currentSession))
    },
    [upsertSession]
  )

  if (!session || !initialState) return null

  return (
    <IngestWizardProvider
      key={session.id}
      initialState={initialState}
      onStateChange={persistWizardState}
    >
      <WizardModalContent
        open={open}
        onClose={onClose}
        autoProcessQueued={autoProcessQueued}
        session={session}
        markProcessingTracking={markProcessingTracking}
        markInterrupted={markInterrupted}
        shouldAttemptPersistedReattach={
          session.lifecycle === "processing" &&
          session.tracking?.mode === "webui-direct" &&
          Boolean(session.tracking?.jobIds?.length)
        }
      />
    </IngestWizardProvider>
  )
}

export default QuickIngestWizardModal
