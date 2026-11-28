import {
  DOCUMENT_WORKSPACE_PATH,
  buildMediaCollectionReviewPath,
} from "@/routes/route-paths"
import { setSetting } from "@/services/settings/registry"
import { MEDIA_REVIEW_SELECTION_SETTING, MEDIA_REVIEW_SELECTION_SNAPSHOT_SETTING } from "@/services/settings/ui-settings"
import { buildKnowledgeMediaScopePath } from "@/utils/knowledge-scope-handoff"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { type ConferenceRetryRequestItem,
  buildConferenceRetryRequestItems,
  hasConferenceCollectionMetadata,
  UNDISPATCHED_PLAN_SOURCE,
  resolveConferenceDuplicatePolicy,
} from "@/services/tldw/conference-collections"
import { resolveIngestUrlType } from "@/services/tldw/media-routing"
import {
  completedIngestJobIndicatesFailure,
  completedIngestJobIndicatesSkipped,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
  extractCompletedIngestJobWarning,
} from "@/services/tldw/ingest-job-results"
import { ACTIVE_INGEST_JOB_STATUSES, INGEST_JOB_STATUS_INTERRUPTED, getQuickIngestJobPollInterval, type IngestJobProgress } from "@/services/tldw/ingest-jobs-orchestrator"
import { quickIngestAuthority, useQuickIngestAuthority } from "@/services/tldw/quick-ingest-authority"
import {
  cancelQuickIngestSession,
  getQuickIngestAnalysisProviderWarning,
  startQuickIngestSession,
  submitQuickIngestBatch,
} from "@/services/tldw/quick-ingest-batch"
import { requestQuickIngestJobCancellation } from "@/services/tldw/quick-ingest-request-scope"
import { reattachQuickIngestSession } from "@/services/tldw/quick-ingest-session-reattach"
import { useConnectionStore } from "@/store/connection"
import { useQuickIngestStore } from "@/store/quick-ingest"
import {
  type PersistedQuickIngestTracking,
  type PersistedWizardQueueItem,
  type QuickIngestSessionLifecycle,
  type QuickIngestSessionRecord,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"
import { ConnectionPhase } from "@/types/connection"
import { isQuickIngestPlaylistPreflightDetail } from "@/utils/quick-ingest-open"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { Button, Modal } from "antd"
import { XCircle } from "lucide-react"
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { browser } from "wxt/browser"
import { useShallow } from "zustand/react/shallow"
import { AddContentStep } from "./QuickIngest/AddContentStep"
import { FloatingProgressWidget } from "./QuickIngest/FloatingProgressWidget"
import {
  IngestWizardProvider,
  type IngestWizardState,
  useIngestWizard
} from "./QuickIngest/IngestWizardContext"
import { IngestWizardStepper } from "./QuickIngest/IngestWizardStepper"
import { ProcessingStep } from "./QuickIngest/ProcessingStep"
import { ReviewStep } from "./QuickIngest/ReviewStep"
import { WizardConfigureStep } from "./QuickIngest/WizardConfigureStep"
import { WizardResultsStep } from "./QuickIngest/WizardResultsStep"
import { WebIngestDraft } from "./QuickIngest/WebIngestDraft"
import {
  DUPLICATE_SKIP_MESSAGE,
  isDbMessageDuplicate,
} from "./QuickIngest/constants"
import {
  DEFAULT_PRESET,
  DEFAULT_PRESETS,
  type PresetMap,
} from "./QuickIngest/presets"
import { getEligibleQueueItems, restoreQueueFile } from "./QuickIngest/queue-items"
import { canRetryWizardResult, getSavedMediaIds } from "./QuickIngest/result-actions"
import type {
  CommonOptions,
  ConferenceBatchMetadata,
  ConferenceItemMetadataOverride,
  DetectedMediaType,
  ItemProgress,
  ItemProgressStatus,
  PlaylistQueueMetadata,
  ReattachedQuickIngestJob,
  TypeDefaults,
  WizardQueueItem,
  WizardResultItem} from "./QuickIngest/types"
import { createReviewDraftsFromResults } from "./hooks/useIngestResults"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type QuickIngestWizardModalProps = {
  oneScreen?: boolean
  open: boolean
  onClose: () => void
  /** When true, automatically skip to processing on mount (compat with old modal). */
  autoProcessQueued?: boolean
  presetMap?: PresetMap
  openRevision?: number
  createNewDraft?: () => QuickIngestSessionRecord
}

type QuickIngestEntryType = "auto" | "html" | "pdf" | "document" | "audio" | "video"

type QuickIngestRequestPayload = {
  entries: Array<{
    id: string
    url: string
    type: QuickIngestEntryType
    defaults?: TypeDefaults
    playlist?: PlaylistQueueMetadata
    conferenceOverride?: ConferenceItemMetadataOverride
    processAgain?: boolean
    sourceUrlEdited?: boolean
  }>
  files: Array<{
    id: string
    name: string
    type?: string
    directFile?: File
    defaults?: TypeDefaults
    conferenceOverride?: ConferenceItemMetadataOverride
  }>
  storeRemote: boolean
  processOnly: boolean
  common: CommonOptions
  advancedValues: Record<string, unknown>
  fileDefaults: TypeDefaults
  conferenceBatchMetadata?: ConferenceBatchMetadata | null
  conferenceRetry?: { collectionId: string; items: ConferenceRetryRequestItem[] }
  planKey?: string
  expectedCollectionId?: string
  __quickIngestSessionId?: string
}

type QuickIngestRuntimeMessage = {
  type: string
  payload?: {
    sessionId?: string
    result?: Partial<WizardResultItem>
    results?: Array<Partial<WizardResultItem>>
    summary?: { interruptedJobIds?: number[] }
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

const FILE_REATTACH_WARNING = "Reattach this file after refresh to process it."
const PERSISTED_REATTACH_POLL_INTERVAL_MS = 1_500
const QUICK_INGEST_MODAL_STYLES = {
  body: {
    padding: "0 16px 16px",
    maxHeight: "calc(100vh - 180px)",
    overflowY: "auto" as const,
  },
}

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

const normalizeWizardResult = (
  item: Partial<WizardResultItem> | null | undefined,
  legacyRuntime = false
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
    ? (item.data as Record<string, unknown>
    )
      : undefined
  const isDuplicate =
    derivedStatus === "ok" &&
    !item.outcome &&
    (
      isDbMessageDuplicate(dataRecord) ||
      completedIngestJobIndicatesSkipped(item.data)
    )
  // The extension worker's legacy cancellation result has only an error message.
  const legacyCancellation = legacyRuntime && /^cancell?ed(?: by user)?\.?$/i.test(String(error || "").trim())
  return {
    id,
    status: derivedStatus,
    outcome:
      isDuplicate
        ? "skipped"
        : derivedStatus === "ok"
          ? item.outcome || "processed"
          : item.outcome || (legacyCancellation ? "cancelled" : "failed"),
    url: item.url,
    fileName: item.fileName,
    type: String(item.type || "item"),
    data: item.data,
    error,
    warning: derivedStatus === "ok"
      ? extractCompletedIngestJobWarning(item.data) || item.warning
      : undefined,
    title: item.title,
    durationMs: item.durationMs,
    mediaId:
      item.mediaId !== undefined
        ? item.mediaId
        : extractCompletedIngestJobMediaId(item.data),
    persisted: item.persisted,
    collectionItemId: item.collectionItemId ?? null,
    retryAttempt: item.retryAttempt ?? null,
    idempotencyKey: item.idempotencyKey ?? null,
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

const preserveConfirmedCancellation = (
  result: WizardResultItem,
  cancelledItemIds: ReadonlySet<string>
): WizardResultItem => cancelledItemIds.has(result.id)
  ? {
      ...result,
      status: "error",
      outcome: "cancelled",
      error: "Cancelled by user.",
      mediaId: null,
      persisted: false,
      warning: undefined,
    }
  : result

const buildTerminalProgress = (
  previous: ItemProgress,
  result: WizardResultItem
): ItemProgress => {
  const cancelled = result.outcome === "cancelled"
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
  outcome: "failed" | "cancelled",
  retryItems?: ConferenceRetryRequestItem[]
): WizardResultItem[] =>
  items.map((item) => {
    const retry = retryItems?.find((entry) => entry.resultId === item.id)
    return {
      id: item.id,
      status: "error",
      outcome,
      url: item.url,
      fileName: item.fileName,
      type: mapDetectedTypeToEntryType(item.detectedType),
      error: message,
      ...(retry
        ? {
            collectionItemId: retry.collectionItemId,
            retryAttempt: retry.retryAttempt,
            idempotencyKey: retry.idempotencyKey
          }
        : {})
    }
  })

const buildQueueFileKey = (item: WizardQueueItem): string | undefined => {
  if (!item.file) return item.fileStub?.key
  const lastModified = Number.isFinite(item.file.lastModified) ? item.file.lastModified : 0
  return `${item.file.name}::${item.file.size}::${lastModified}`
}

const buildPersistedQueueItems = (
  items: WizardQueueItem[]
): PersistedWizardQueueItem[] =>
  items.map((item) => ({
    id: item.id,
    kind: item.kind || (item.url ? "url" : "file"),
    fileOrigin: item.fileOrigin,
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
    playlist: item.playlist,
    conferenceOverride: item.conferenceOverride,
    processAgain: item.processAgain,
    sourceUrlEdited: item.sourceUrlEdited,
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
  const originalIndex = tracking?.jobIds?.indexOf(jobId) ?? -1
  return resolveTrackedQueueItems(items, tracking)[originalIndex >= 0 ? originalIndex : index]
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
  const jobIdToCollectionItemId = Object.entries(tracking.jobIdToCollectionItemId ?? {})
    .map(([jobId, itemId]) => `${jobId}:${String(itemId || "").trim()}`)
    .sort()

  return [
    mode,
    sessionId,
    batchIds.join(","),
    jobIds.join(","),
    itemIds.join(","),
    jobIdToItemId.join(","),
    jobIdToCollectionItemId.join(","),
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
        playlist: item.playlist,
        conferenceOverride: item.conferenceOverride,
        processAgain: item.processAgain,
        sourceUrlEdited: item.sourceUrlEdited,
      }
    }

    const warnings = (item.validation.warnings ?? []).filter(warning => warning !== FILE_REATTACH_WARNING)
    if (item.fileOrigin !== "paste") warnings.push(FILE_REATTACH_WARNING)

    return {
      id: item.id,
      kind: "file",
      fileOrigin: item.fileOrigin,
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
      playlist: item.playlist,
      conferenceOverride: item.conferenceOverride,
      processAgain: item.processAgain,
      sourceUrlEdited: item.sourceUrlEdited,
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
  if (state.currentStep < 4 && state.processingState.status === "idle" && !state.processingState.correctionItemIds?.length) {
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
  const attemptedIds = new Set([
    ...state.processingState.perItemProgress.map(item => item.id),
    ...state.results.map(item => item.id),
  ])

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
    totalCount: attemptedIds.size || state.queueItems.length,
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
  // A restored correction has already visited Results and can submit its retry.
  highestStep: session.currentStep <= 3 && session.results.length ? 5 : session.currentStep,
  queueItems: hydrateQueueItems(session.queueItems),
  selectedPreset: session.selectedPreset,
  customBasePreset: session.customBasePreset,
  presetConfig: session.presetConfig,
  customOptions: session.customOptions,
  playlistPreflightSeed: isQuickIngestPlaylistPreflightDetail(session.openDetail)
    ? session.openDetail
    : null,
  firstSourceAddMode: session.firstSourceAddMode ?? null,
  processingState: session.processingState,
  results: session.results,
  conferenceBatchMetadata: session.conferenceBatchMetadata ?? null,
  isMinimized:
    session.visibility === "hidden" && session.lifecycle === "processing",
})

const buildOpenDetailPatch = (
  state: IngestWizardState,
  session: QuickIngestSessionRecord
): QuickIngestSessionRecord["openDetail"] => {
  if (state.playlistPreflightSeed) {
    return state.playlistPreflightSeed
  }
  if (isQuickIngestPlaylistPreflightDetail(session.openDetail)) {
    return null
  }
  return session.openDetail ?? null
}

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
    conferenceBatchMetadata: state.conferenceBatchMetadata,
    processingState: state.processingState,
    results: state.results,
    openDetail: buildOpenDetailPatch(state, session),
    firstSourceAddMode: state.firstSourceAddMode ?? null,
    badge: {
      queueCount:
        lifecycle === "draft"
          ? getEligibleQueueItems(state.queueItems).length
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

const buildWizardPersistenceSignature = (
  patch: Partial<QuickIngestSessionRecord>
): string => {
  const resultSummary = patch.resultSummary
  return JSON.stringify({
    ...patch,
    completedAt: patch.completedAt == null ? null : true,
    resultSummary: resultSummary
      ? {
          ...resultSummary,
          attemptedAt: resultSummary.attemptedAt == null ? null : true,
          completedAt: resultSummary.completedAt == null ? null : true,
        }
      : resultSummary,
  })
}

const cancelQuickIngestSessionBestEffort = (
  request: Parameters<typeof cancelQuickIngestSession>[0]
): void => {
  void cancelQuickIngestSession(request).catch((error) => {
    console.warn("[QuickIngest] Failed to cancel session.", {
      sessionId: request.sessionId,
      error,
    })
  })
}

const mapReattachedJobStatusToProgress = (
  status: string,
  result?: unknown
): ItemProgressStatus => {
  switch (status) {
    case "unknown":
      return "unknown"
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
    const retry = tracking?.retryItems?.find(
      (entry) => entry.resultId === item?.id
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
      outcome: isDuplicate
        ? ("skipped" as const)
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
      warning: resultStatus === "ok" ? extractCompletedIngestJobWarning(job.result) : undefined,
      collectionItemId:
        tracking?.jobIdToCollectionItemId?.[String(job.jobId)] ??
        retry?.collectionItemId ??
        null,
      retryAttempt: retry?.retryAttempt ?? null,
      idempotencyKey: retry?.idempotencyKey ?? null,
      title: job.result?.title ?? null,
      data: job.result,
      message: isDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined
    }
  })

const buildTrackedItemProgress = (
  id: string,
  job: IngestJobProgress & { result?: unknown; error?: string }
): ItemProgress => {
  const status = mapReattachedJobStatusToProgress(job.status, job.result)
  const terminal = status === "complete" || status === "failed" || status === "cancelled"
  return {
    id,
    status,
    progressPercent: terminal ? 100 : job.progressPercent ?? 0,
    currentStage: terminal ? "" : job.progressMessage || "",
    estimatedRemaining: 0,
    error: status === "failed"
      ? job.error || extractCompletedIngestJobError(job.result) || "Failed"
      : undefined,
  }
}

const buildProgressFromReattachedJobs = (
  items: WizardQueueItem[],
  jobs: ReattachedQuickIngestJob[],
  tracking: PersistedQuickIngestTracking | undefined,
  cancelledItems: ReadonlySet<string>
): ItemProgress[] =>
  jobs.map((job, index) => {
    const item = resolveQueueItemForReattachedJob(
      items,
      tracking,
      job.sourceItemId,
      job.jobId,
      index
    )
    const id = item?.id || `reattached-${job.jobId}`
    return buildTrackedItemProgress(id, cancelledItems.has(id) ? { status: "cancelled" } : job)
  })

const buildQuickIngestPayload = (
  items: WizardQueueItem[],
  conferenceBatchMetadata: ConferenceBatchMetadata | null,
  options: QuickIngestRequestPayload["common"] & {
    storeRemote: boolean
    reviewBeforeStorage: boolean
    advancedValues?: Record<string, unknown>
    typeDefaults: TypeDefaults
  }
): QuickIngestRequestPayload => {
  const validItems = getEligibleQueueItems(items)
  const entries = validItems
    .filter((item): item is WizardQueueItem & { url: string } => Boolean(item.url))
    .map((item) => ({
      id: item.id,
      url: item.url,
      type: mapDetectedTypeToEntryType(item.detectedType),
      defaults: buildDefaultsForQueueItem(item, options.typeDefaults),
      playlist: item.playlist,
      conferenceOverride: item.conferenceOverride,
      processAgain: item.processAgain,
      sourceUrlEdited: item.sourceUrlEdited,
    }))

  const files = validItems
    .filter((item): item is WizardQueueItem & { file: File } => Boolean(item.file))
    .map((item) => ({
      id: item.id,
      name: item.file.name,
      type: item.file.type || undefined,
      directFile: item.file,
      defaults: buildDefaultsForQueueItem(item, options.typeDefaults),
      conferenceOverride: item.conferenceOverride,
    }))

  return {
    entries,
    files,
    storeRemote: options.storeRemote,
    processOnly: options.reviewBeforeStorage || !options.storeRemote,
    common: {
      perform_analysis: options.perform_analysis,
      perform_chunking: options.perform_chunking,
      overwrite_existing: options.overwrite_existing,
      chunking_mode: options.chunking_mode,
      auto_chunking_goal: options.auto_chunking_goal,
      auto_chunking_use_llm: options.auto_chunking_use_llm,
    },
    advancedValues: options.advancedValues ?? {},
    fileDefaults: options.typeDefaults,
    conferenceBatchMetadata,
  }
}

// ---------------------------------------------------------------------------
// Inner modal content (must be inside IngestWizardProvider)
// ---------------------------------------------------------------------------

type WizardModalContentProps = {
  oneScreen: boolean
  open: boolean
  onClose: () => void
  autoProcessQueued?: boolean
  session: QuickIngestSessionRecord
  markProcessingTracking: (tracking: PersistedQuickIngestTracking) => void
  markInterrupted: (reason?: string) => void
  showSession: () => void
  replaceWithNewDraft: () => QuickIngestSessionRecord
  shouldAttemptPersistedReattach: boolean
  reviewWasOpened: () => boolean
  markReviewOpened: () => void
  releaseReviewOpened: () => void
}

const WizardModalContent: React.FC<WizardModalContentProps> = ({
  oneScreen,
  open,
  onClose,
  autoProcessQueued = false,
  session,
  markProcessingTracking,
  markInterrupted,
  showSession,
  replaceWithNewDraft,
  shouldAttemptPersistedReattach,
  reviewWasOpened,
  markReviewOpened,
  releaseReviewOpened,
}) => {
  const [operation] = useState(() => quickIngestAuthority.capture())
  const { t } = useTranslation(["option"])
  const {
    state,
    minimize,
    restore,
    skipToProcessing,
    cancelItem,
    updateItemProgress,
    updateProcessingState,
    setResults,
    setQueueItems,
    goToStep,
    goNext,
  } = useIngestWizard()
  const { currentStep, queueItems, processingState, presetConfig, results } = state
  const [mobileViewport, setMobileViewport] = useState<React.CSSProperties>()
  useEffect(() => {
    const viewport = window.visualViewport
    if (!oneScreen || !open || state.isMinimized || !viewport) return
    const measure = () => setMobileViewport({
      "--qi-visible-height": `${viewport.height}px`,
      "--qi-visible-top": `${viewport.offsetTop}px`,
    } as React.CSSProperties)
    measure()
    viewport.addEventListener("resize", measure)
    viewport.addEventListener("scroll", measure)
    return () => {
      viewport.removeEventListener("resize", measure)
      viewport.removeEventListener("scroll", measure)
    }
  }, [oneScreen, open, state.isMinimized])
  const [analysisProviderWarning, setAnalysisProviderWarning] = useState<
    string | null
  >(null)
  const connectionState = useConnectionStore((store) => store.state)
  const checkConnection = useConnectionStore((store) => store.checkOnce)
  const activeSessionIdRef = useRef<string | null>(null)
  const resultsRef = useRef(results)
  const hasStartedRunRef = useRef(false)
  const restoredPlanRun = useRef(currentStep === 4 && session.lifecycle === "processing" && session.tracking?.planKey
    ? session.tracking : null)
  const runStartedAtRef = useRef<number | null>(null)
  const cancelledSessionIdsRef = useRef<Set<string>>(new Set())
  const cancelRequestedRef = useRef(false)
  const stopRequestRef = useRef<string | null>(null)
  const [stopRequestInFlight, setStopRequestInFlight] = useState(false)
  const cancelledItemIdsRef = useRef(new Set<string>())
  const cancelledJobIdsRef = useRef(new Set<number>())
  const [jobStatusUnavailable, setJobStatusUnavailable] = useState(false)
  const [retryIds, setRetryIds] = useState<string[] | null>(() =>
    currentStep === 4 && processingState.perItemProgress.length
      ? processingState.perItemProgress.map(item => item.id)
      : currentStep <= 3 ? processingState.correctionItemIds ?? null : null
  )
  const retryRequestsRef = useRef<ConferenceRetryRequestItem[]>(session.tracking?.retryItems ?? (session.tracking?.durableMode === "durable_collection" ? buildConferenceRetryRequestItems(results) : []))
  const retryCollectionIdRef = useRef<string | undefined>(session.tracking?.collectionId)
  const validQueueItems = useMemo(
    () =>
      getEligibleQueueItems(queueItems).filter(item => (!retryIds || retryIds.includes(item.id)) && !results.some(result => result.id === item.id && result.status === "ok")),
    [queueItems, retryIds, results]
  )
  const trackedQueueItems = useMemo(
    () => resolveTrackedQueueItems(queueItems, session.tracking),
    [queueItems, session.tracking]
  )
  const replacementItems = useMemo(() => {
    if (!presetConfig.storeRemote || presetConfig.reviewBeforeStorage) return []
    return validQueueItems.filter(item => {
      if (item.url && resolveIngestUrlType(item.url, mapDetectedTypeToEntryType(item.detectedType)) === "html") return false
      const policy = resolveConferenceDuplicatePolicy(item.playlist?.duplicateStatus, item.conferenceOverride?.duplicatePolicy, item.processAgain)
      return policy.shouldSubmitJob && (presetConfig.common.overwrite_existing || policy.forceOverwrite)
    })
  }, [presetConfig, validQueueItems])
  const requiresReplacementReview = replacementItems.length > 0
  const trackedJobIdsByItem = useMemo(() => {
    const tracking = session.tracking
    const result = new Map<string, number[]>()
    if (tracking?.mode === "webui-direct") {
      normalizeTrackedJobIds(tracking).forEach((jobId, index) => {
        const item = resolveQueueItemForReattachedJob(queueItems, tracking, undefined, jobId, index)
        if (!item) return
        const jobIds = result.get(item.id) ?? []
        jobIds.push(jobId)
        result.set(item.id, jobIds)
      })
    }
    return result
  }, [queueItems, session.tracking])
  const initialTrackedQueueItemsRef = useRef(trackedQueueItems)
  const itemProgressRef = useRef(processingState.perItemProgress)
  const remainingPlanProgress = useRef<ItemProgress[]>([])
  const checkingPlan = useRef<AbortController | null>(null)
  useEffect(() => () => checkingPlan.current?.abort(), [])
  itemProgressRef.current = processingState.perItemProgress
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
  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t],
  )
  const isOnlineForIngest =
    connectionState.offlineBypass === true ||
    (
      connectionState.isConnected &&
      connectionState.phase === ConnectionPhase.CONNECTED
    )
  const isCheckingConnection =
    connectionState.isChecking ||
    connectionState.phase === ConnectionPhase.SEARCHING
  const connectionRecoveryMessage = useMemo(() => {
    if (isCheckingConnection) {
      return qi(
        "wizard.offline.checkingDescription",
        "Checking your tldw server connection before processing."
      )
    }
    if (connectionState.phase === ConnectionPhase.UNCONFIGURED) {
      return qi(
        "wizard.offline.unconfiguredDescription",
        "Configure your tldw server under Settings -> tldw server before processing."
      )
    }
    if (connectionState.lastError) {
      return qi(
        "wizard.offline.errorDescription",
        "Cannot reach your tldw server. {{error}}",
        { error: connectionState.lastError }
      )
    }
    return qi(
      "wizard.offline.description",
      "Reconnect to your tldw server before processing. You can still add URLs and configure queued items."
    )
  }, [
    connectionState.lastError,
    connectionState.phase,
    isCheckingConnection,
    qi,
  ])
  const handleRetryConnection = useCallback(() => {
    void checkConnection()
  }, [checkConnection])

  useEffect(() => {
    if (!analysisProviderWarning) return
    const providerWarning = getQuickIngestAnalysisProviderWarning({
      common: presetConfig.common,
      advancedValues: presetConfig.advancedValues,
    })
    if (!providerWarning) {
      setAnalysisProviderWarning(null)
    }
  }, [analysisProviderWarning, presetConfig.advancedValues, presetConfig.common])

  useEffect(() => {
    resultsRef.current = results
  }, [results])

  useEffect(() => {
    initialTrackedQueueItemsRef.current = trackedQueueItems
  }, [trackedQueueItems])

  useEffect(() => {
    persistedTrackingRef.current = session.tracking
    const sessionId = String(session.tracking?.sessionId || "").trim()
    if (sessionId && !hasStartedRunRef.current && processingState.status === "running") {
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
  }, [session.tracking, processingState.status])

  useEffect(() => {
    if (!shouldAttemptPersistedReattach) {
      activeReattachSignatureRef.current = ""
    }
  }, [shouldAttemptPersistedReattach])

  // Track recently ingested documents for the DocumentPickerModal
  const addRecentlyIngestedDocs = useQuickIngestStore(s => s.addRecentlyIngestedDocs)
  const recordedIngestRef = useRef(false)

  useEffect(() => {
    if (!operation.isCurrent()) return
    if (state.currentStep !== 5) { recordedIngestRef.current = false; return }
    if (recordedIngestRef.current) return
    recordedIngestRef.current = true
    const savedMediaIds = state.results.filter(item => item.status === "ok" && item.mediaId != null)
      .map(item => Number(item.mediaId)).filter(id => Number.isFinite(id) && id > 0)
    if (savedMediaIds.length > 0) {
      window.dispatchEvent(new CustomEvent("tldw:quick-ingest-complete", { detail: {
        mediaIds: [...new Set(savedMediaIds)], authorityKey: operation.authorityKey,
        isCurrent: operation.isCurrent
      } }))
    }
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
  }, [state.currentStep, state.results, addRecentlyIngestedDocs, operation])

  useEffect(() => {
    if (!open || !state.isMinimized) return
    restore()
  }, [open, restore, state.isMinimized])

  // Auto-process on mount if autoProcessQueued is set and there are queued items
  const autoProcessedRef = useRef(false)

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

  useEffect(() => {
    const persistedTracking = persistedTrackingRef.current
    const reattachQueueItems = initialTrackedQueueItemsRef.current
    // Recovery polls must not race the live executor's complete mixed result set.
    // A freshly mounted/reloaded session has no local run and may reattach.
    if (hasStartedRunRef.current) return
    if (!persistedReattachSignature || !persistedTracking) return
    if (activeReattachSignatureRef.current === persistedReattachSignature) return
    activeReattachSignatureRef.current = persistedReattachSignature

    const startedAt = persistedTracking.startedAt
    if (typeof startedAt === "number" &&
      Number.isFinite(startedAt)) {
      runStartedAtRef.current = startedAt
    }
    const sessionId = String(persistedTracking.sessionId || "").trim()
    if (sessionId) {
      activeSessionIdRef.current = sessionId
    }

    let cancelled = false
    const pollingStartedAt = runStartedAtRef.current || Date.now()
    const scheduleNextPoll = (statusUnavailable = false) => {
      persistedReattachTimerRef.current = window.setTimeout(() => {
        void pollPersistedTracking().catch(() => { /* Owner changes abandon client polling. */ })
      }, getQuickIngestJobPollInterval(pollingStartedAt, PERSISTED_REATTACH_POLL_INTERVAL_MS, statusUnavailable))
    }
    const pollPersistedTracking = async () => {
      const currentTracking = persistedTrackingRef.current || persistedTracking
      const confirmed = new Set(cancelledJobIdsRef.current)
      const snapshot = confirmed.size
        ? await reattachQuickIngestSession(currentTracking, operation, confirmed)
        : await reattachQuickIngestSession(currentTracking, operation)
      if (cancelled || cancelRequestedRef.current || !operation.isCurrent()) return
      if (confirmed.size !== cancelledJobIdsRef.current.size) {
        scheduleNextPoll()
        return
      }
      setJobStatusUnavailable(snapshot.statusUnavailable === true)

      const latestTracking = persistedTrackingRef.current || currentTracking
      const reportedProgress = buildProgressFromReattachedJobs(
        reattachQueueItems,
        snapshot.jobs,
        latestTracking,
        cancelledItemIdsRef.current
      )
      const progress = new Map(itemProgressRef.current.map(item => [item.id, item]))
      for (const [index, jobId] of normalizeTrackedJobIds(latestTracking).entries()) {
        const item = resolveQueueItemForReattachedJob(reattachQueueItems, latestTracking, undefined, jobId, index)
        if (!item || progress.has(item.id)) continue
        const unknown = buildTrackedItemProgress(item.id, { status: "unknown" })
        const previous = resultsRef.current.find(result => result.id === item.id)
        progress.set(item.id, previous ? buildTerminalProgress(unknown, previous) : unknown)
      }
      for (const item of reportedProgress) progress.set(item.id, item)
      const perItemProgress = [...progress.values()]
      const elapsed =
        typeof startedAt === "number" && Number.isFinite(startedAt)
          ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
          : initialElapsedRef.current

      const reattachedResults = snapshot.lifecycle === "partial_failure" && snapshot.jobs.length === 0
        ? buildFailureResults(
            reattachQueueItems,
            snapshot.errorMessage || "Quick ingest failed.",
            "failed",
            latestTracking.retryItems
          )
        : buildResultsFromReattachedJobs(
            reattachQueueItems,
            snapshot.jobs.filter((job) => !ACTIVE_INGEST_JOB_STATUSES.has(job.status)),
            latestTracking
          )

      const confirmedResults = reattachedResults.map((result) =>
        preserveConfirmedCancellation(result, cancelledItemIdsRef.current)
      )
      const mergedResults = mergeWizardResults(resultsRef.current, confirmedResults)
      if (confirmedResults.length) {
        resultsRef.current = mergedResults
        setResults(mergedResults)
      }
      const resultIds = new Set(mergedResults.map(result => result.id))
      const allOutcomesKnown = perItemProgress.length > 0 && perItemProgress.every(item => resultIds.has(item.id))
      if (snapshot.lifecycle === "processing" && !allOutcomesKnown) {
        updateProcessingState({ status: "running", perItemProgress, elapsed, estimatedRemaining: 0 })
        scheduleNextPoll(snapshot.statusUnavailable)
        return
      }

      // A reload may leave an in-flight upload without a durable job ID.
      const interrupted = !allOutcomesKnown
      updateProcessingState({
        status:
          interrupted || snapshot.lifecycle === "partial_failure"
            ? "error"
            : mergedResults.length > 0 && mergedResults.every(result => result.outcome === "cancelled")
              ? "cancelled"
              : "complete",
        perItemProgress,
        elapsed,
        estimatedRemaining: 0,
      })
      hasStartedRunRef.current = false
      activeSessionIdRef.current = null
      if (interrupted) {
        markInterrupted(
          snapshot.errorMessage || "Quick ingest could not reconnect to live job status."
        )
      }
      goNext()
    }

    void pollPersistedTracking().catch(() => { /* Owner changes abandon client polling. */ })

    return () => {
      cancelled = true
      // A replacement effect (including StrictMode replay) must own a fresh poll.
      if (activeReattachSignatureRef.current === persistedReattachSignature) {
      activeReattachSignatureRef.current = ""
    }
      if (persistedReattachTimerRef.current != null) {
        window.clearTimeout(persistedReattachTimerRef.current)
        persistedReattachTimerRef.current = null
      }
    }
  }, [
    goNext,
    markInterrupted,
    persistedReattachSignature,
    operation,
    setResults,
    updateProcessingState,
  ])

  const applyResults = useCallback(
    (incoming: WizardResultItem[]) => {
      const extractOnly = oneScreen && !presetConfig.storeRemote && !presetConfig.reviewBeforeStorage
      const confirmedResults = incoming.map((item) => {
        const result = preserveConfirmedCancellation(item, cancelledItemIdsRef.current)
        return extractOnly && result.data != null
          ? { ...result, omitDataOnReload: true } : result
      })
      const next = mergeWizardResults(resultsRef.current, confirmedResults)
      resultsRef.current = next
      setResults(next)
      const progressMap = new Map(
        itemProgressRef.current.map((item) => [item.id, item])
      )
      for (const result of confirmedResults) {
        const previous = progressMap.get(result.id) || buildTrackedItemProgress(result.id, { status: "pending" })
        updateItemProgress(buildTerminalProgress(previous, result))
      }
      return next
    },
    [oneScreen, presetConfig.storeRemote, presetConfig.reviewBeforeStorage, setResults, updateItemProgress]
  )

  const finalizeStatusInterruption = useCallback((confirmedResults: WizardResultItem[]) => {
    applyResults(confirmedResults)
    markInterrupted(INGEST_JOB_STATUS_INTERRUPTED)
    updateProcessingState({ status: "error", estimatedRemaining: 0 })
    hasStartedRunRef.current = false
    activeSessionIdRef.current = null
    goNext()
  }, [applyResults, goNext, markInterrupted, updateProcessingState])

  const finalizeRun = useCallback(
    (
      nextStatus: "complete" | "cancelled" | "error",
      incomingResults: WizardResultItem[]
    ) => {
      syncElapsed()
      const resolvedIds = new Set([...resultsRef.current, ...incomingResults].map(item => item.id))
      applyResults([...incomingResults, ...buildFailureResults(validQueueItems.filter(item => !resolvedIds.has(item.id)), "The server returned no result for this input. Review the source and try again.", "failed", retryRequestsRef.current.length ? retryRequestsRef.current : persistedTrackingRef.current?.retryItems)])
      const unconfirmed = remainingPlanProgress.current.some(item => !resultsRef.current.some(result => result.id === item.id))
      if (unconfirmed) markInterrupted(INGEST_JOB_STATUS_INTERRUPTED)
      updateProcessingState({ status: unconfirmed ? "error" : nextStatus, estimatedRemaining: 0 })
      hasStartedRunRef.current = false
      activeSessionIdRef.current = null
      goNext()
    },
    [applyResults, goNext, markInterrupted, syncElapsed, updateProcessingState, validQueueItems]
  )

  const finalizeFailure = useCallback(
    (message: string, outcome: "failed" | "cancelled") => {
      const trackedEligibleItems = getEligibleQueueItems(trackedQueueItems).filter(item => retryIds == null || retryIds.includes(item.id))
      const fallbackItems =
        trackedEligibleItems.length > 0 ? trackedEligibleItems : validQueueItems
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
        outcome,
        retryRequestsRef.current.length ? retryRequestsRef.current : persistedTrackingRef.current?.retryItems
      )
      finalizeRun(outcome === "cancelled" ? "cancelled" : "error", fallbackResults)
    },
    [finalizeRun, trackedQueueItems, validQueueItems, retryIds]
  )

  const handleCancelItem = useCallback(async (itemId: string) => {
    const sessionId = activeSessionIdRef.current
    const jobIds = trackedJobIdsByItem.get(itemId)
    if (!jobIds?.length || !operation.isCurrent() || cancelRequestedRef.current) return false

    const settled = await Promise.allSettled(
      jobIds.map((jobId) => requestQuickIngestJobCancellation(operation, jobId))
    )
    const confirmed = settled.every((result) => result.status === "fulfilled" && result.value)
    if (confirmed && operation.isCurrent() && sessionId === activeSessionIdRef.current && !cancelRequestedRef.current) {
      cancelledItemIdsRef.current.add(itemId)
      jobIds.forEach((jobId) => cancelledJobIdsRef.current.add(jobId))
      cancelItem(itemId)
    }
    return confirmed
  }, [cancelItem, operation, trackedJobIdsByItem])

  const handleCancelAll = useCallback(() => {
    if (!operation.isCurrent() || cancelRequestedRef.current) return
    const persistedTracking = persistedTrackingRef.current
    const sessionId = activeSessionIdRef.current || ""
    const restoredJobs = !hasStartedRunRef.current && persistedTracking?.mode === "webui-direct" && Boolean(persistedTracking.jobIds?.length)
    if (restoredJobs || (sessionId && (sessionId.startsWith("qi-direct-") || persistedTracking?.mode === "webui-direct"))) {
      if (stopRequestRef.current === sessionId) return
      stopRequestRef.current = sessionId
      setStopRequestInFlight(true)
      updateProcessingState({ bulkStop: "requested" })
      const finishRequest = (acknowledged: boolean) => {
        if (!operation.isCurrent() || stopRequestRef.current !== sessionId) return
        stopRequestRef.current = null
        setStopRequestInFlight(false)
        if (!acknowledged && sessionId === (activeSessionIdRef.current || "")) {
          updateProcessingState({ bulkStop: "unconfirmed" })
        }
      }
      const batchIds = resolveTrackingBatchIds(persistedTracking)
      if (restoredJobs && (!sessionId || !batchIds.length)) {
        const unresolved = [...trackedJobIdsByItem.keys()].filter(id =>
          !cancelledItemIdsRef.current.has(id) && !resultsRef.current.some(result => result.id === id)
        )
        void Promise.all(unresolved.map(handleCancelItem)).then(
          outcomes => finishRequest(outcomes.length > 0 && outcomes.every(Boolean)), () => finishRequest(false)
        )
      } else {
        void cancelQuickIngestSession({
          requestScope: operation.requestScope, signal: operation.signal, assertCurrent: operation.assertCurrent,
          sessionId, batchIds, reason: "user_cancelled",
        }).then(response => finishRequest(response?.ok === true), () => finishRequest(false))
      }
      return
    }

    cancelRequestedRef.current = true
    if (sessionId) {
      cancelledSessionIdsRef.current.add(sessionId)
      cancelQuickIngestSessionBestEffort({
        requestScope: operation.requestScope, signal: operation.signal, assertCurrent: operation.assertCurrent,
        sessionId,
        batchIds: resolveTrackingBatchIds(persistedTracking),
        reason: "user_cancelled",
      })
    }

    finalizeFailure("Cancelled by user.", "cancelled")
  }, [finalizeFailure, handleCancelItem, operation, trackedJobIdsByItem, updateProcessingState])

  const markRunActive = useCallback(() => {
    runStartedAtRef.current = Date.now()
    updateProcessingState({ bulkStop: undefined, perItemProgress: [...remainingPlanProgress.current, ...validQueueItems.map(item => ({
      id: item.id, status: "processing" as const, progressPercent: 0,
      currentStage: "Waiting for server result", estimatedRemaining: 0,
    }))] })
  }, [updateProcessingState, validQueueItems])

  const handleRuntimeMessage = useCallback(
    (message: QuickIngestRuntimeMessage) => {
      if (!operation.isCurrent() || !message || typeof message.type !== "string") return
      const sessionId = String(message.payload?.sessionId || "").trim()
      if (!sessionId || sessionId !== String(activeSessionIdRef.current || "").trim()) {
        return
      }
      if (cancelledSessionIdsRef.current.has(sessionId)) {
        return
      }

      if (message.type === "tldw:quick-ingest/progress") {
        const result = normalizeWizardResult(message.payload?.result, true)
        if (result) {
          applyResults([result])
        }
        return
      }

      if (message.type === "tldw:quick-ingest/completed") {
        const normalizedResults = (message.payload?.results || [])
          .map((item) => normalizeWizardResult(item, true))
          .filter((item): item is WizardResultItem => Boolean(item))
        if (message.payload?.summary?.interruptedJobIds?.length) {
          finalizeStatusInterruption(normalizedResults)
          return
        }
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
    [applyResults, finalizeFailure, finalizeRun, finalizeStatusInterruption, operation]
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
    if (!operation.isCurrent() || hasStartedRunRef.current || validQueueItems.length === 0) return
    hasStartedRunRef.current = true
    activeSessionIdRef.current = null
    cancelledItemIdsRef.current.clear()
    cancelledJobIdsRef.current.clear()
    setJobStatusUnavailable(false)
    const attemptedIds = new Set(validQueueItems.map((item) => item.id))
    const retained = resultsRef.current.filter(
      (item) => !attemptedIds.has(item.id) || item.status === "ok"
    )
    resultsRef.current = retained
    setResults(retained)

    try {
      try {
        await tldwClient.initialize()
      } catch {
        // Best effort; background proxy handles auth for direct runtimes.
      }
      if (!operation.isCurrent() || cancelRequestedRef.current) return

      const requestPayload = buildQuickIngestPayload(
        validQueueItems,
        state.conferenceBatchMetadata,
        {
          ...presetConfig.common,
          storeRemote: presetConfig.storeRemote,
          reviewBeforeStorage: presetConfig.reviewBeforeStorage,
          advancedValues: presetConfig.advancedValues,
          typeDefaults: presetConfig.typeDefaults,
        }
      )
      if (!operation.isCurrent() || cancelRequestedRef.current) return
      if (retryRequestsRef.current.length && retryCollectionIdRef.current) {
        requestPayload.conferenceRetry = { collectionId: retryCollectionIdRef.current, items: retryRequestsRef.current }
      }

      const resumedPlan = session.tracking?.planKey === session.id
      if (oneScreen && requestPayload.storeRemote && !requestPayload.processOnly && (resumedPlan ||
          (!requestPayload.conferenceRetry && hasConferenceCollectionMetadata(state.conferenceBatchMetadata, validQueueItems)))) {
        requestPayload.planKey = session.id
        if (resumedPlan) requestPayload.expectedCollectionId = session.tracking?.collectionId
      }

      const providerWarning = getQuickIngestAnalysisProviderWarning({
        common: requestPayload.common,
        advancedValues: requestPayload.advancedValues,
      })
      if (providerWarning) {
        setAnalysisProviderWarning(
          qi(
            "analysisProvider.required",
            "Choose an analysis provider before running ingest analysis."
          )
        )
        updateProcessingState({
          status: "idle",
          perItemProgress: [],
          elapsed: 0,
          estimatedRemaining: 0,
        })
        hasStartedRunRef.current = false
        activeSessionIdRef.current = null
        restore()
        showSession()
        goToStep(2)
        return
      }

      markRunActive()

      const startAck = await startQuickIngestSession({ ...requestPayload, requestScope: operation.requestScope, signal: operation.signal, assertCurrent: operation.assertCurrent })
      if (!operation.isCurrent()) return
      if (cancelRequestedRef.current) {
        const cancelledSessionId = String(startAck?.sessionId || "").trim()
        if (startAck?.ok && cancelledSessionId) {
          cancelledSessionIdsRef.current.add(cancelledSessionId)
          if (!cancelledSessionId.startsWith("qi-direct-")) {
            cancelQuickIngestSessionBestEffort({
        requestScope: operation.requestScope, signal: operation.signal, assertCurrent: operation.assertCurrent,
              sessionId: cancelledSessionId,
              reason: "user_cancelled",
            })
          }
        }
        return
      }
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
        planKey: requestPayload.planKey,
        collectionId: requestPayload.expectedCollectionId,
        ...(requestPayload.conferenceRetry
          ? {
              collectionId: requestPayload.conferenceRetry.collectionId,
              retryItems: requestPayload.conferenceRetry.items,
              plannedItemIds: requestPayload.conferenceRetry.items.map(
                (item) => item.collectionItemId
              ),
              durableMode: "durable_collection"
            }
          : {}),
        startedAt: runStartedAtRef.current || Date.now()
      })

      if (!sessionId.startsWith("qi-direct-")) {
        return
      }

      const response = await submitQuickIngestBatch({
        ...requestPayload,
        requestScope: operation.requestScope, signal: operation.signal, assertCurrent: operation.assertCurrent,
        __quickIngestSessionId: sessionId,
        isJobCancellationConfirmed: (itemId) => cancelledItemIdsRef.current.has(itemId),
        onStopUnconfirmed: () => {
          if (operation.isCurrent() && activeSessionIdRef.current === sessionId) updateProcessingState({ bulkStop: "unconfirmed" })
        },
        onJobStatusUnavailable: (unavailable) => {
          if (operation.isCurrent() && !cancelRequestedRef.current) setJobStatusUnavailable(unavailable)
        },
        onJobProgress: (itemId, progress) => {
          if (!operation.isCurrent() || cancelRequestedRef.current || cancelledItemIdsRef.current.has(itemId)) return
          updateItemProgress(buildTrackedItemProgress(itemId, progress))
        },
        onItemResult: (item) => {
          if (!operation.isCurrent() || cancelRequestedRef.current) return
          const result = normalizeWizardResult(item)
          if (result) applyResults([result])
        },
        onTrackingMetadata: (tracking) => {
          if (!operation.isCurrent() || cancelRequestedRef.current) return
          markProcessingTracking({
            ...tracking,
            sessionId,
            mode: "webui-direct",
            startedAt: tracking.startedAt || runStartedAtRef.current || Date.now(),
          })
        },
      })

      if (
        !operation.isCurrent() ||
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

      if (response.interruptedJobIds?.length || response.interruptedItemIds?.length) {
        finalizeStatusInterruption(normalizedResults)
        return
      }

      if (normalizedResults.length === 0) {
        finalizeFailure("Ingest request finished without item results.", "failed")
        return
      }

      finalizeRun("complete", normalizedResults)
    } catch (error) {
      if (!operation.isCurrent() || cancelRequestedRef.current) return
      finalizeFailure(
        error instanceof Error ? error.message : "Quick ingest failed.",
        "failed"
      )
    }
  }, [
    applyResults,
    finalizeStatusInterruption,
    updateItemProgress,
    finalizeFailure,
    finalizeRun,
    setResults,
    markRunActive,
    presetConfig.advancedValues,
    presetConfig.common,
    presetConfig.reviewBeforeStorage,
    presetConfig.storeRemote,
    presetConfig.typeDefaults,
    state.conferenceBatchMetadata,
    oneScreen,
    session.id,
    session.tracking?.planKey,
    session.tracking?.collectionId,
    goToStep,
    restore,
    showSession,
    markProcessingTracking,
    qi,
    updateProcessingState,
    validQueueItems,
    operation,
  ])

  useEffect(() => {
    if (currentStep !== 4 || processingState.status !== "running" || session.lifecycle === "interrupted") return
    if (session.lifecycle === "processing" && session.tracking) {
      const canReattachDirectJobs =
        session.tracking.mode === "webui-direct" &&
        Boolean(session.tracking.jobIds?.length)
      if (session.tracking.mode !== "webui-direct" || canReattachDirectJobs) {
        return
      }
      if (restoredPlanRun.current && !hasStartedRunRef.current) {
        // Let hydration publish its initial running snapshot before recording the interruption.
        queueMicrotask(() => {
          if (!restoredPlanRun.current || !operation.isCurrent()) return
          restoredPlanRun.current = null
          finalizeStatusInterruption([])
        })
        return
      }
      if (processingState.bulkStop && !hasStartedRunRef.current) {
        finalizeStatusInterruption([])
        return
      }
    }
    void startRun()
  }, [currentStep, processingState.status, processingState.bulkStop, session.lifecycle, session.tracking, startRun, finalizeStatusInterruption, operation])

  // Modal title with item count
  const modalTitle = useMemo(() => {
    const base = qi("wizard.title", "Quick Ingest")
    if (queueItems.length > 0 && currentStep <= 3) {
      return `${base} (${queueItems.length})`
    }
    return base
  }, [qi, queueItems.length, currentStep])

  const [processingPrompt, setProcessingPrompt] = useState<"leave" | "stop" | null>(null)
  const stayButton = useRef<HTMLButtonElement>(null)
  const requestStop = useCallback(() => setProcessingPrompt("stop"), [])
  useEffect(() => {
    if (!isProcessingActive || !open || state.isMinimized) setProcessingPrompt(null)
  }, [isProcessingActive, open, state.isMinimized])
  useEffect(() => {
    if (processingPrompt) stayButton.current?.focus()
  }, [processingPrompt])
  const remaining = processingState.perItemProgress.filter(item =>
    item.status !== "complete" && item.status !== "failed" && item.status !== "cancelled" && item.status !== "unknown"
  ).length

  // Close handler with confirmation when processing
  const handleCloseAttempt = useCallback(() => {
    if (oneScreen && isProcessingActive) {
      setProcessingPrompt("leave")
    } else if (isProcessingActive) {
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
                handleCancelAll()
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
  }, [oneScreen, handleCancelAll, isProcessingActive, qi, minimize, onClose])

  const beginRetry = useCallback((ids: string[]) => {
    if (!ids.length) return
    checkingPlan.current?.abort()
    checkingPlan.current = null
    setRetryIds(ids)
    remainingPlanProgress.current = session.lifecycle === "interrupted" && session.tracking?.planKey
      ? itemProgressRef.current.filter(item => !ids.includes(item.id) && !resultsRef.current.some(result => result.id === item.id)) : []
    const selected = resultsRef.current.filter(item => ids.includes(item.id))
    const requests = session.tracking?.durableMode === "durable_collection"
      ? buildConferenceRetryRequestItems(selected) : []
    retryRequestsRef.current = requests
    retryCollectionIdRef.current = session.tracking?.collectionId
    const retained = resultsRef.current.filter(item => !ids.includes(item.id))
    resultsRef.current = retained
    setResults(retained)
    cancelRequestedRef.current = false
    hasStartedRunRef.current = false
    activeSessionIdRef.current = null
    runStartedAtRef.current = null
    // A reload before acknowledgment must never reattach the previous attempt's jobs.
    markProcessingTracking({
      mode: "webui-direct",
      sessionId: `qi-direct-retry-${crypto.randomUUID()}`,
      planKey: session.tracking?.planKey,
      ...(session.tracking?.planKey ? { collectionId: retryCollectionIdRef.current } : {}),
      startedAt: Date.now(),
      ...(retryCollectionIdRef.current && requests.length ? {
        collectionId: retryCollectionIdRef.current, retryItems: requests, durableMode: "durable_collection" as const,
      } : {}),
    })
    updateProcessingState({
      status: "running",
      bulkStop: undefined,
      correctionItemIds: undefined,
      perItemProgress: [...remainingPlanProgress.current, ...ids.map(id => ({
        id, status: "queued" as const, progressPercent: 0, currentStage: "", estimatedRemaining: 0,
      }))],
      elapsed: 0,
      estimatedRemaining: 0,
    })
    goToStep(4)
  }, [goToStep, markProcessingTracking, session.lifecycle, session.tracking, setResults, updateProcessingState])

  // Quick-process callback for AddContentStep (skip to processing with defaults)
  const handleQuickProcess = useCallback(() => {
    if (!operation.isCurrent()) return
    if (!isOnlineForIngest || isCheckingConnection) return
    const providerWarning = getQuickIngestAnalysisProviderWarning({
      common: presetConfig.common,
      advancedValues: presetConfig.advancedValues,
    })
    if (providerWarning) {
      setAnalysisProviderWarning(
        qi(
          "analysisProvider.required",
          "Choose an analysis provider before running ingest analysis."
        )
      )
      if (oneScreen) return
      if (currentStep === 1) {
        goNext()
      } else {
        goToStep(2)
      }
      return
    }
    setAnalysisProviderWarning(null)
    if (retryIds || resultsRef.current.length) {
      beginRetry(validQueueItems.map(item => item.id))
    } else {
      skipToProcessing()
    }
  }, [
    currentStep,
    goNext,
    goToStep,
    isCheckingConnection,
    isOnlineForIngest,
    presetConfig.advancedValues,
    presetConfig.common,
    qi,
    skipToProcessing,
    oneScreen,
    retryIds,
    beginRetry,
    validQueueItems,
    operation,
  ])

  useEffect(() => {
    if (!open || !autoProcessQueued) {
      autoProcessedRef.current = false
    }
  }, [autoProcessQueued, open])

  useEffect(() => {
    if (
      autoProcessQueued &&
      !autoProcessedRef.current &&
      validQueueItems.length > 0 &&
      isOnlineForIngest &&
      !isCheckingConnection
    ) {
      autoProcessedRef.current = true
      // An automatic launch must still let the user resolve prerequisites or review replacements.
      if (!oneScreen || !requiresReplacementReview) {
        handleQuickProcess()
      }
    }
  }, [
    autoProcessQueued,
    oneScreen,
    requiresReplacementReview,
    handleQuickProcess,
    isCheckingConnection,
    isOnlineForIngest,
    validQueueItems.length,
  ])

  const handleRetryItems = useCallback(
    (itemIds: string[], requests?: ConferenceRetryRequestItem[]) => {
      if (!operation.isCurrent() || !isOnlineForIngest || isCheckingConnection) return
      const requested = new Set(requests?.map(item => item.resultId) ?? itemIds)
      const failed = resultsRef.current.filter(item => requested.has(item.id) && canRetryWizardResult(item, queueItems))
      beginRetry(failed.map(item => item.id))
    },
    [beginRetry, isCheckingConnection, isOnlineForIngest, operation, queueItems]
  )

  const handleReattach = useCallback((id: string, file: File) => {
    if (!operation.isCurrent()) return
    const mismatch = qi("wizard.results.reattachMismatch", "Choose the original file with the same name and size.")
    setQueueItems(queueItems.map(item => item.id === id ? restoreQueueFile(item, file, "upload", mismatch) : item))
  }, [operation, qi, queueItems, setQueueItems])

  const handleCorrectItems = useCallback((ids: string[], repairSource = false) => {
    if (!operation.isCurrent()) return
    if (repairSource) {
      setQueueItems(queueItems.map(item => ids.includes(item.id)
        ? { ...item, validation: { ...item.validation, valid: false } }
        : item))
    }
    setRetryIds(ids)
    updateProcessingState({ correctionItemIds: ids })
    retryRequestsRef.current = session.tracking?.durableMode === "durable_collection" ? buildConferenceRetryRequestItems(resultsRef.current.filter(item => ids.includes(item.id))) : []
    retryCollectionIdRef.current = session.tracking?.collectionId
    goToStep(repairSource ? 1 : 2)
  }, [goToStep, operation, queueItems, session.tracking, setQueueItems, updateProcessingState])

  // Navigation callbacks for WizardResultsStep CTAs
  const navigateInPlace = useNavigate()
  const [navigationError, setNavigationError] = useState<string | null>(null)
  const navigationMountedRef = useRef(false)
  useEffect(() => {
    navigationMountedRef.current = true
    return () => { navigationMountedRef.current = false }
  }, [])
  const navigate = useCallback((route: string, onFailure?: () => void) => {
    if (!operation.isCurrent()) return
    if (window.location.pathname.endsWith("/sidepanel.html")) {
      setNavigationError(null)
      void (async () => {
        try {
          const url = browser.runtime.getURL(`/options.html#${route}`)
          if (browser.tabs?.create) await browser.tabs.create({ url })
          else if (!window.open(url, "_blank")) throw new Error("Workspace tab was blocked")
          // The captured operation includes the session generation; the mount
          // check also fences a replaced host with the same stored session.
          if (navigationMountedRef.current && operation.isCurrent()) onClose()
        } catch {
          onFailure?.()
          if (navigationMountedRef.current && operation.isCurrent()) {
            setNavigationError(qi("workspaceOpenFailed", "Could not open the full-page workspace. Try the action again."))
          }
        }
      })()
      return
    }
    try {
      navigateInPlace(route)
      onClose()
    } catch (error) {
      onFailure?.()
      throw error
    }
  }, [navigateInPlace, onClose, operation, qi])
  const [reviewError, setReviewError] = useState<string | null>(null)
  const [reviewRetry, setReviewRetry] = useState(0)
  const reviewSourceKey = useMemo(() => JSON.stringify(results.filter(item => item.status === "ok").map(item => item.id).sort()), [results])
  const [draftBatch, setDraftBatch] = useState<{ batchId: string; count: number; sourceKey: string } | null>(null)
  const reviewCreationRef = useRef<{ sourceKey: string; promise: ReturnType<typeof createReviewDraftsFromResults> } | null>(null)
  useEffect(() => {
    if (!open || currentStep !== 5 || !presetConfig.reviewBeforeStorage || (!oneScreen && reviewWasOpened()) || draftBatch?.sourceKey === reviewSourceKey || !results.some(item => item.status === "ok")) return
    if (reviewCreationRef.current?.sourceKey !== reviewSourceKey) {
      setReviewError(null)
      const files = new Map<string, File>()
      for (const item of queueItems) if (item.file) files.set(item.id, item.file)
      const promise = createReviewDraftsFromResults({
        results, files, operation,
        batchId: "quick-ingest-review:" + session.id,
        processingOptions: {
          perform_analysis: Boolean(presetConfig.common.perform_analysis),
          perform_chunking: Boolean(presetConfig.common.perform_chunking),
          overwrite_existing: Boolean(presetConfig.common.overwrite_existing),
          advancedValues: { ...presetConfig.advancedValues }
        }
      })
      reviewCreationRef.current = { sourceKey: reviewSourceKey, promise }
    }
    // Reuse the same write across rerenders and StrictMode, but only the
    // currently open wizard may publish its completion to the UI.
    let cancelled = false
    void reviewCreationRef.current.promise.then(batch => {
      if (cancelled || !operation.isCurrent() || !batch) return
      if (oneScreen) {
        setDraftBatch({ batchId: batch.batchId, count: batch.draftIds.length, sourceKey: reviewSourceKey })
        return
      }
      if (reviewWasOpened()) return
      markReviewOpened()
      navigate("/content-review?batch=" + encodeURIComponent(batch.batchId), releaseReviewOpened)
    }).catch(error => {
      if (!cancelled && operation.isCurrent()) setReviewError(error instanceof Error ? error.message : "Failed to save review drafts.")
    })
    return () => { cancelled = true }
  }, [currentStep, draftBatch, navigate, oneScreen, open, operation, presetConfig, queueItems, results, reviewRetry, reviewSourceKey, session.id, reviewWasOpened, markReviewOpened, releaseReviewOpened])

  const handleSearchKnowledge = useCallback(
    (mediaIds: number[]) => {
      if (!operation.isCurrent()) return
      navigate(buildKnowledgeMediaScopePath(mediaIds))
    },
    [navigate, operation],
  )

  const handleIngestMore = useCallback(() => {
    if (!operation.isCurrent()) return
    replaceWithNewDraft()
  }, [replaceWithNewDraft, operation])

  const handleOpenWorkspace = useCallback(
    (item: WizardResultItem) => {
      if (!operation.isCurrent()) return
      const mediaId = item.mediaId
      if (mediaId != null) {
        navigate(`${DOCUMENT_WORKSPACE_PATH}?open=${mediaId}`)
      } else {
        navigate(DOCUMENT_WORKSPACE_PATH)
      }
    },
    [navigate, operation]
  )

  const handleOpenMedia = useCallback(
    (item: WizardResultItem) => {
      if (!operation.isCurrent()) return
      const mediaId = item.mediaId
      const mediaPath = mediaId != null
        ? `/media?id=${encodeURIComponent(String(mediaId))}`
        : "/media"
      navigate(mediaPath)
    },
    [navigate, operation]
  )

  const handleReviewSavedItems = useCallback(
    async (ids: Array<string | number>) => {
      if (!operation.isCurrent()) return
      const current = getSavedMediaIds(resultsRef.current)
      const requested = new Set(ids.map(String))
      const selected = current.filter((id) => requested.has(String(id)))
      if (!selected.length) return
      // This single record owns the handoff; a stale snapshot cannot relabel another owner's IDs.
      await setSetting(MEDIA_REVIEW_SELECTION_SNAPSHOT_SETTING, {
      version: 1, authorityKey: operation.authorityKey, selectedIds: selected,
    })
      if (!operation.isCurrent()) return
      try {
      await setSetting(MEDIA_REVIEW_SELECTION_SETTING, selected)
    } catch {
      // Legacy readers receive a best-effort mirror; owned readers use the snapshot above.
      console.warn("[media-review] Compatibility selection mirror could not be written.")
    }
      if (!operation.isCurrent()) return
      navigate("/media-multi")
    },
    [navigate, operation]
  )

  const handleOpenCollection = useCallback(
    (collectionId: string) => {
      if (!operation.isCurrent()) return
      const collectionPath = buildMediaCollectionReviewPath(collectionId)
      navigate(collectionPath)
    },
    [navigate, operation]
  )

  const [planCheckFailed, setPlanCheckFailed] = useState(false)
  const [missingPlanSources, setMissingPlanSources] = useState<string[]>([])
  const checkInterruptedPlan = useCallback(async (ids: string[]) => {
    if (checkingPlan.current || !operation.isCurrent() || session.tracking?.planKey !== session.id) return
    const controller = new AbortController()
    const isCurrentCheck = () => {
      const live = useQuickIngestSessionStore.getState().session
      return !controller.signal.aborted && operation.isCurrent() && live?.id === session.id && live.lifecycle === "interrupted" &&
        live.tracking?.sessionId === session.tracking?.sessionId
    }
    checkingPlan.current = controller
    setPlanCheckFailed(false)
    setMissingPlanSources([])
    try {
      const selected = queueItems.filter(item => ids.includes(item.id))
      const payload = buildQuickIngestPayload(selected, state.conferenceBatchMetadata, {
        ...presetConfig.common, storeRemote: true, reviewBeforeStorage: false, typeDefaults: presetConfig.typeDefaults,
      })
      // A receipt lookup needs only the stable source identity, including after native bytes were released.
      for (const item of selected) {
        if (!item.url && !payload.files.some(file => file.id === item.id)) payload.files.push({ id: item.id,
          name: item.fileName || "upload", type: item.mimeType, conferenceOverride: item.conferenceOverride })
      }
      const trackingUpdates: PersistedQuickIngestTracking[] = []
      const response = await submitQuickIngestBatch({ ...payload, planKey: session.id, checkPlanOnly: true, expectedCollectionId: session.tracking.collectionId,
        __quickIngestSessionId: session.tracking.sessionId, requestScope: operation.requestScope,
        signal: AbortSignal.any([operation.signal, controller.signal]), assertCurrent: operation.assertCurrent,
        onTrackingMetadata: tracking => { trackingUpdates.push(tracking) },
      })
      if (!isCurrentCheck()) return
      setMissingPlanSources(response.missingPlanSourceIds ?? [])
      setPlanCheckFailed(Boolean(response.statusCheckUnavailable || !response.ok))
      for (const tracking of trackingUpdates) markProcessingTracking(tracking)
      const confirmed = (response.results ?? []).map(item => normalizeWizardResult(
        (item.data as { code?: string } | undefined)?.code === UNDISPATCHED_PLAN_SOURCE
          ? { ...item, error: qi("web.planReadySuggestion", "Retry to submit it to the original collection.") } : item
      )).filter((item): item is WizardResultItem => Boolean(item))
      applyResults(confirmed)
      if (response.ok && !response.interruptedItemIds?.length && !response.interruptedJobIds?.length) {
        updateProcessingState({ status: "complete", estimatedRemaining: 0 })
      } else {
        markInterrupted(INGEST_JOB_STATUS_INTERRUPTED)
      }
    } catch {
      if (isCurrentCheck()) setPlanCheckFailed(true)
    } finally {
      if (checkingPlan.current === controller) checkingPlan.current = null
    }
  }, [applyResults, markProcessingTracking, markInterrupted, operation, presetConfig, qi, queueItems, session.id, session.tracking,
    state.conferenceBatchMetadata, updateProcessingState])

  const interruption = useMemo(() => {
    if (!oneScreen || session.lifecycle !== "interrupted") return
    const activeIds = new Set([
      ...normalizeTrackedItemIds(session.tracking),
      ...trackedJobIdsByItem.keys(),
      ...processingState.perItemProgress.map(progress => progress.id),
    ])
    const sources = queueItems.filter(item =>
      activeIds.has(item.id) && !results.some(result => result.id === item.id)
    ).map(item => ({
      id: item.id,
      name: item.conferenceOverride?.title || item.fileName || item.url || item.id,
      jobIds: trackedJobIdsByItem.get(item.id) ?? [],
      lastProgress: processingState.perItemProgress.find(progress => progress.id === item.id),
    }))
    const tracking = session.tracking
    const setupSources = !tracking?.collectionId ? sources : sources.filter(source => missingPlanSources.includes(source.id))
    return {
      sources,
      onCheckMedia: () => navigate("/media"),
      checkError: planCheckFailed ? qi("web.planCheckFailed", "Could not refresh the collection source status.") : null,
      onResumeSetup: tracking?.planKey === session.id && setupSources.length ? () => {
        if (!operation.isCurrent() || checkingPlan.current) return
        setPlanCheckFailed(false)
        const ids = new Set(setupSources.map(source => source.id))
        const eligible = getEligibleQueueItems(queueItems).filter(item => ids.has(item.id))
        const missing = queueItems.filter(item => ids.has(item.id) && !eligible.some(ready => ready.id === item.id))
        if (missing.length) applyResults(buildFailureResults(missing, qi("web.reattachToResumeSetup", "Reattach the original file to retry collection setup."), "failed")
          .map(item => ({ ...item, data: { code: UNDISPATCHED_PLAN_SOURCE } })))
        if (eligible.length) beginRetry(eligible.map(item => item.id))
        else if (setupSources.length === sources.length) updateProcessingState({ status: "complete", estimatedRemaining: 0 })
      } : undefined,
      onCheckStatus: tracking?.planKey === session.id && sources.length
        ? () => { void checkInterruptedPlan(sources.map(source => source.id)) }
        : tracking?.mode === "webui-direct" && sources.some(source => source.jobIds.length > 0)
        ? () => {
          if (!operation.isCurrent()) return
          markProcessingTracking(tracking)
          updateProcessingState({ status: "running" })
          goToStep(4)
        } : undefined,
    }
  }, [oneScreen, session.lifecycle, session.tracking, trackedJobIdsByItem, processingState.perItemProgress,
    queueItems, results, navigate, operation, markProcessingTracking, updateProcessingState, goToStep, session.id, checkInterruptedPlan,
    planCheckFailed, missingPlanSources, applyResults, beginRetry, qi])

  // Render the current step
  const stepContent = useMemo(() => {
    if (oneScreen && currentStep <= 3) {
      return <WebIngestDraft processingItems={validQueueItems} replacementItems={replacementItems}
        visible={open && !state.isMinimized}
        isOnlineForIngest={isOnlineForIngest} isCheckingConnection={isCheckingConnection}
        connectionRecoveryMessage={connectionRecoveryMessage} onRetryConnection={handleRetryConnection}
        onProcess={handleQuickProcess} />
    }
    switch (currentStep) {
      case 1:
        return (
          <AddContentStep
            isOnlineForIngest={isOnlineForIngest}
            isCheckingConnection={isCheckingConnection}
            connectionRecoveryMessage={connectionRecoveryMessage}
            onRetryConnection={handleRetryConnection}
            onQuickProcess={handleQuickProcess}
          />
        )
      case 2:
        return (
          <WizardConfigureStep
            processingItems={validQueueItems}
            isStepVisible={open && !state.isMinimized && currentStep === 2}
            analysisProviderWarning={analysisProviderWarning}
            focusAnalysisProvider={Boolean(analysisProviderWarning)}
          />
        )
      case 3:
        return (
          <ReviewStep
            processingItems={validQueueItems}
            isOnlineForIngest={isOnlineForIngest}
            isCheckingConnection={isCheckingConnection}
            connectionRecoveryMessage={connectionRecoveryMessage}
            onRetryConnection={handleRetryConnection}
          />
        )
      case 4:
        return (
          <ProcessingStep
            oneScreen={oneScreen}
            onCancelAll={oneScreen ? requestStop : handleCancelAll}
            jobStatusUnavailable={jobStatusUnavailable}
            stopRequestInFlight={stopRequestInFlight}
            onCancelItem={handleCancelItem}
            trackedJobIdsByItem={trackedJobIdsByItem}
            localItemCancellation={!oneScreen && session.tracking?.mode !== "webui-direct"}
            onMinimize={onClose}
          />
        )
      case 5:
        return (
          <>
            {!oneScreen && session.lifecycle === "interrupted" && (
              <div role="alert">{INGEST_JOB_STATUS_INTERRUPTED}</div>
            )}
            {navigationError && <div role="alert">{navigationError}</div>}
            {reviewError && (
              <div role="alert">
                <p>{reviewError}</p>
                <Button onClick={() => { reviewCreationRef.current = null; setReviewRetry(value => value + 1) }}>
                  {qi("reviewDraftsRetry", "Retry saving review drafts")}
                </Button>
              </div>
            )}
            <WizardResultsStep
            oneScreen={oneScreen}
            interruption={interruption}
            reviewDrafts={oneScreen && draftBatch?.sourceKey === reviewSourceKey && draftBatch.count > 0 ? {
              count: draftBatch.count,
              onOpen: () => navigate("/content-review?batch=" + encodeURIComponent(draftBatch.batchId)),
            } : undefined}
            onClose={onClose}
            onIngestMore={handleIngestMore}
            onOpenMedia={handleOpenMedia}
            onSearchKnowledge={handleSearchKnowledge}
            onOpenWorkspace={handleOpenWorkspace}
            onOpenCollection={handleOpenCollection}
            onRetryItems={handleRetryItems}
            onCorrectItems={handleCorrectItems}
            onReattach={handleReattach}
            onReviewSavedItems={handleReviewSavedItems}
          />
          </>
        )
      default:
        return null
    }
  }, [
    connectionRecoveryMessage,
    oneScreen,
    draftBatch,
    reviewSourceKey,
    replacementItems,
    requestStop,
    interruption,
    navigate,
    currentStep,
    reviewError,
    navigationError,
    qi,
    handleCancelAll,
    handleCancelItem,
    jobStatusUnavailable,
    stopRequestInFlight,
    session.lifecycle,
    session.tracking?.mode,
    trackedJobIdsByItem,
    handleOpenMedia,
    handleOpenCollection,
    handleRetryItems,
    handleCorrectItems,
    handleReattach,
    handleReviewSavedItems,
    handleOpenWorkspace,
    handleIngestMore,
    handleQuickProcess,
    handleRetryConnection,
    handleSearchKnowledge,
    isCheckingConnection,
    isOnlineForIngest,
    onClose,
    open,
    analysisProviderWarning,
    state.isMinimized,
    validQueueItems,
  ])

  return (
    <>
      <Modal
        open={open && !state.isMinimized}
        onCancel={handleCloseAttempt}
        title={oneScreen ? qi("wizard.title", "Quick Ingest") : modalTitle}
        footer={null}
        width={oneScreen ? 680 : 800}
        centered={oneScreen}
        classNames={oneScreen ? { wrapper: "qi-web-wrap", container: "qi-web-surface", header: "qi-web-header", body: "qi-web-body" } : undefined}
        className={oneScreen ? "quick-ingest-modal qi-web-modal" : "quick-ingest-modal quick-ingest-wizard-modal"}
        getContainer={false}
        styles={oneScreen ? { wrapper: mobileViewport } : QUICK_INGEST_MODAL_STYLES}
      >
        {oneScreen ? stepContent : <>
          <IngestWizardStepper />
          <div className="min-h-[300px]">{stepContent}</div>
        </>}
      </Modal>

      {oneScreen && <Modal
        open={open && isProcessingActive && Boolean(processingPrompt)}
        title={processingPrompt === "stop" ? qi("web.stopTitle", "Stop the remaining import?") : qi("web.leaveTitle", "Leave this import?")}
        centered closable={false} width={440} footer={null}
        className="qi-processing-prompt"
        onCancel={() => setProcessingPrompt(null)}
        afterOpenChange={visible => { if (visible) stayButton.current?.focus() }}
      >
        <p className="text-sm text-text">
          {remaining > 0 ? remaining === 1 ? qi("web.oneActive", "1 source is still active.") :
            qi("web.sourcesActive", "{{count}} sources are still active.", { count: remaining })
            : qi("web.waitingForStatus", "Waiting for the latest source status.")}
        </p>
        <p className="mt-1 text-sm text-text-muted">{processingPrompt === "stop"
          ? qi("web.stopHelp", "Saved items stay in Media. Quick Ingest will check which jobs stopped.")
          : qi("web.leaveHelp", "You can keep working here while your import continues in the background.")}</p>
        <div className="mt-5 flex flex-wrap justify-end gap-2">
          <Button ref={stayButton} onClick={() => setProcessingPrompt(null)}>
            {processingPrompt === "stop" ? qi("web.keepImporting", "Keep importing") : qi("web.stayHere", "Stay here")}
          </Button>
          <Button danger onClick={() => {
            if (processingPrompt === "leave") requestStop()
            else {
              setProcessingPrompt(null)
              handleCancelAll()
            }
          }}>{qi("web.stopRemaining", "Stop remaining sources")}</Button>
          {processingPrompt === "leave" && <Button type="primary" onClick={() => {
            setProcessingPrompt(null)
            minimize()
            onClose()
          }}>{qi("web.continueBackground", "Continue in background")}</Button>}
        </div>
      </Modal>}

      {/* Floating progress widget (renders via portal when minimized) */}
      <FloatingProgressWidget />
    </>
  )
}

// ---------------------------------------------------------------------------
// Exported modal component
// ---------------------------------------------------------------------------

export const QuickIngestWizardModal: React.FC<QuickIngestWizardModalProps> = ({
  oneScreen = !isExtensionRuntime(),
  open,
  onClose,
  autoProcessQueued = false,
  presetMap = DEFAULT_PRESETS,
  openRevision = 0,
  createNewDraft,
}) => {
  const authorityKey = useQuickIngestAuthority()
  const {
    session,
    generation,
    upsertSession,
    markProcessingTracking,
    markInterrupted,
    createDraftSession,
    showSession,
    replaceWithNewDraft,
  } =
    useQuickIngestSessionStore(
      useShallow((store) => ({
        session: store.session,
        generation: store.generation,
        upsertSession: store.upsertSession,
        markProcessingTracking: store.markProcessingTracking,
        markInterrupted: store.markInterrupted,
        createDraftSession: store.createDraftSession,
        showSession: store.showSession,
        replaceWithNewDraft: store.replaceWithNewDraft,
      }))
    )

  // Keep attached files in this modal only; persisted stubs still require reattach after reload.
  const liveQueueRef = useRef<{
    authorityKey: string
    sessionId: string
    items: WizardQueueItem[]
  } | null>(null)
  const initialState = useMemo(() => {
    if (!session) {
      liveQueueRef.current = null
      return undefined
    }
    const initial = buildInitialWizardState(session)
    const live = liveQueueRef.current
    if (
      !live ||
      live.authorityKey !== authorityKey ||
      live.sessionId !== session.id
    ) {
      liveQueueRef.current = null
      return initial
    }
    const attached = new Map(
      live.items.filter((item) => item.file).map((item) => [item.id, item])
    )
    return {
      ...initial,
      queueItems: initial.queueItems.map((item) => {
        const current = attached.get(item.id)
        return current
          ? { ...item, file: current.file, validation: current.validation }
          : item
      })
    }
  }, [authorityKey, session])
  const boundSessionId = session?.id
  const reviewHandoffKey = JSON.stringify([authorityKey, boundSessionId, generation])
  // Keep the receipt outside the provider's openRevision remount.
  const publishedReviewRef = useRef<string | null>(null)
  const reviewWasOpened = useCallback(
    () => publishedReviewRef.current === reviewHandoffKey,
    [reviewHandoffKey]
  )
  const markReviewOpened = useCallback(() => {
    publishedReviewRef.current = reviewHandoffKey
  }, [reviewHandoffKey])
  const releaseReviewOpened = useCallback(() => {
    if (publishedReviewRef.current === reviewHandoffKey) publishedReviewRef.current = null
  }, [reviewHandoffKey])
  const lastPersistedSignatureRef = useRef<{
    sessionId: string
    signature: string
  } | null>(null)

  useEffect(() => {
    if (!authorityKey || !open || session) return
    createDraftSession({
      selectedPreset: DEFAULT_PRESET,
      customBasePreset: DEFAULT_PRESET,
      presetConfig: presetMap[DEFAULT_PRESET],
      customOptions: {},
    })
  }, [authorityKey, createDraftSession, open, presetMap, session])

  const persistWizardState = useCallback(
    (state: IngestWizardState) => {
      const { session: currentSession, authorityKey: currentAuthorityKey } =
        useQuickIngestSessionStore.getState()
      if (
        !authorityKey || !currentSession || currentSession.id !== boundSessionId ||
        currentAuthorityKey !== authorityKey
      ) return
      liveQueueRef.current = {
        authorityKey,
        sessionId: currentSession.id,
        items: state.queueItems,
      }
      const patch = buildSessionPatchFromWizardState(state, currentSession)
      const signature = buildWizardPersistenceSignature(patch)
      // React may replay queued reducer updates after this synchronous Zustand
      // write rerenders the parent. Persist each semantic snapshot only once.
      if (
        lastPersistedSignatureRef.current?.sessionId === currentSession.id &&
        lastPersistedSignatureRef.current.signature === signature
      ) {
        return
      }
      lastPersistedSignatureRef.current = {
        sessionId: currentSession.id,
        signature,
      }
      upsertSession(patch)
    },
    [authorityKey, boundSessionId, upsertSession]
  )

  if (!authorityKey || !session || !initialState) return null

  const providerKey = `${session.id}:${openRevision}`

  return (
    <IngestWizardProvider
      key={providerKey}
      initialState={initialState}
      onStateChange={persistWizardState}
      presetMap={presetMap}
    >
      <WizardModalContent
        oneScreen={oneScreen}
        open={open}
        onClose={onClose}
        autoProcessQueued={autoProcessQueued}
        session={session}
        markProcessingTracking={markProcessingTracking}
        markInterrupted={markInterrupted}
        showSession={showSession}
        replaceWithNewDraft={createNewDraft ?? replaceWithNewDraft}
        reviewWasOpened={reviewWasOpened}
        markReviewOpened={markReviewOpened}
        releaseReviewOpened={releaseReviewOpened}
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
