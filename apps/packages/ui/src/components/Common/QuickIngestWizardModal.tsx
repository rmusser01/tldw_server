import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Modal, Button } from "antd"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { XCircle } from "lucide-react"
import { useShallow } from "zustand/react/shallow"
import { browser } from "wxt/browser"
import {
  applyPlaylistReviewRequiredState,
  buildPlaylistIngestRunRequest,
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
  getQuickIngestAnalysisProviderWarning,
  queryQuickIngestSession,
  startQuickIngestSession,
  submitQuickIngestBatch,
} from "@/services/tldw/quick-ingest-batch"
import type { PlaylistReviewRequiredRecoveryItem } from "@/services/tldw/playlist-ingest"
import { reattachQuickIngestSession } from "@/services/tldw/quick-ingest-session-reattach"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  completedIngestJobIndicatesFailure,
  completedIngestJobIndicatesSkipped,
  extractCompletedIngestJobError,
  extractCompletedIngestJobMediaId,
} from "@/services/tldw/ingest-job-results"
import {
  DOCUMENT_WORKSPACE_PATH,
  buildMediaCollectionReviewPath,
} from "@/routes/route-paths"
import {
  type PersistedWizardQueueItem,
  type QuickIngestSessionLifecycle,
  type QuickIngestSessionRecord,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase } from "@/types/connection"
import type {
  CommonOptions,
  ConferenceBatchMetadata,
  ConferenceItemMetadataOverride,
  DetectedMediaType,
  ItemProgress,
  ItemProgressStatus,
  PlaylistQueueMetadata,
  PersistedQuickIngestTracking,
  ReattachedQuickIngestJob,
  TypeDefaults,
  WizardQueueItem,
  WizardResultItem,
} from "./QuickIngest/types"
import {
  DUPLICATE_SKIP_MESSAGE,
  isDbMessageDuplicate,
} from "./QuickIngest/constants"
import { isQuickIngestPlaylistPreflightDetail } from "@/utils/quick-ingest-open"
import {
  DEFAULT_PRESET,
  DEFAULT_PRESETS,
  type PresetMap,
} from "./QuickIngest/presets"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type QuickIngestWizardModalProps = {
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
  }>
  files: Array<{
    id: string
    name: string
    type?: string
    data: number[]
    defaults?: TypeDefaults
    conferenceOverride?: ConferenceItemMetadataOverride
  }>
  storeRemote: boolean
  processOnly: boolean
  common: CommonOptions
  advancedValues: Record<string, unknown>
  fileDefaults: TypeDefaults
  conferenceBatchMetadata?: ConferenceBatchMetadata | null
  conferenceItemMetadata?: Record<
    string,
    {
      playlist?: PlaylistQueueMetadata
      conferenceOverride?: ConferenceItemMetadataOverride
    }
  >
  pendingRunRequest: IngestWizardState["pendingRunRequest"]
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
    occurrenceId?: string
    jobId?: number | null
    status?: string
    runId?: string
    recoverable?: boolean
    reviewRequired?: PlaylistReviewRequiredRecoveryItem[]
  }
}

const AMBIGUOUS_RUN_CREATE_RECOVERY_MESSAGE =
  "Quick ingest may have created a server run before this page reloaded. Reconnect from a confirmed run instead of starting it again."

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

const isCancelledError = (value: unknown) =>
  RESULT_CANCELLED_STATUS_TOKENS.some((token) =>
    String(value || "").trim().toLowerCase().includes(token)
  )

const normalizeResultOutcomeToken = (
  value: unknown
): WizardResultItem["outcome"] | undefined => {
  const outcome = String(value || "").trim().toLowerCase()
  switch (outcome) {
    case "ingested":
    case "processed":
    case "submit_failed":
    case "failed":
    case "cancelled":
      return outcome
    case "canceled":
      return "cancelled"
    case "skipped":
    case "skipped_existing":
    case "included_existing":
      return "skipped"
    default:
      return undefined
  }
}

const normalizeResultOutcome = (
  itemOutcome: unknown,
  dataOutcome: unknown
): WizardResultItem["outcome"] | undefined =>
  normalizeResultOutcomeToken(itemOutcome) ||
  normalizeResultOutcomeToken(dataOutcome)

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
  const normalizedOutcome = normalizeResultOutcome(
    item.outcome,
    dataRecord?.outcome
  )
  const isDuplicate =
    derivedStatus === "ok" &&
    !normalizedOutcome &&
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
          ? normalizedOutcome === "ingested" ||
            normalizedOutcome === "processed" ||
            normalizedOutcome === "skipped"
            ? normalizedOutcome
            : "processed"
          : normalizedOutcome === "cancelled" || isCancelledError(error)
            ? "cancelled"
            : normalizedOutcome === "submit_failed"
              ? "submit_failed"
              : "failed",
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
    sourceRef: item.sourceRef,
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
    playlist: item.playlist,
    playlistReview: item.playlistReview,
    conferenceOverride: item.conferenceOverride,
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
  jobId: number | null,
  index: number
): WizardQueueItem | undefined => {
  const mappedItemId =
    String(sourceItemId || "").trim() ||
    (jobId === null ? undefined : tracking?.jobIdToItemId?.[String(jobId)])
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
  const runId = String(tracking.runId || "").trim()
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
    runId,
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
    const sourceKind = item.sourceRef?.kind
    const isFileItem =
      sourceKind === "file_stub" ||
      (!sourceKind &&
        (item.kind === "file" || (!item.url && Boolean(item.fileName || item.name))))
    if (!isFileItem) {
      const url =
        item.sourceRef?.kind === "direct_url"
          ? item.sourceRef.url
          : item.url || item.playlist?.sourceUrl
      return {
        id: item.id,
        sourceRef: item.sourceRef,
        kind: "url",
        url,
        detectedType: item.detectedType,
        icon: item.icon,
        fileSize: item.fileSize,
        mimeType: item.mimeType,
        validation: item.validation,
        playlist: item.playlist,
        playlistReview: item.playlistReview,
        conferenceOverride: item.conferenceOverride,
      }
    }

    const warnings = Array.from(
      new Set([...(item.validation.warnings ?? []), FILE_REATTACH_WARNING])
    )

    return {
      id: item.id,
      sourceRef: item.sourceRef,
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
      playlist: item.playlist,
      playlistReview: item.playlistReview,
      conferenceOverride: item.conferenceOverride,
      fileStub: item.fileStub || {
        key: item.key,
        lastModified: item.lastModified,
      },
    }
  })

const deriveLifecycleFromWizardState = (
  state: IngestWizardState,
  existingSession: Pick<QuickIngestSessionRecord, "lifecycle" | "tracking">
): QuickIngestSessionLifecycle => {
  const existingLifecycle = existingSession.lifecycle
  if (state.currentStep < 4 && state.processingState.status === "idle") {
    return "draft"
  }

  if (
    existingLifecycle === "interrupted" &&
    existingSession.tracking?.mode === "extension-runtime" &&
    state.processingState.status === "running"
  ) {
    return "interrupted"
  }

  if (state.processingState.status === "running") {
    return "processing"
  }

  if (state.processingState.status === "cancelled") {
    return "cancelled"
  }

  if (state.processingState.status === "error") {
    if (existingSession.tracking?.submissionState === "creating_run") {
      return "interrupted"
    }
    if (existingLifecycle === "interrupted") {
      return "interrupted"
    }
    const trackedItemIds = normalizeTrackedItemIds(existingSession.tracking)
    const resultItemIds = new Set(state.results.map((result) => result.id))
    if (
      existingSession.tracking?.runId &&
      (trackedItemIds.length === 0 ||
        trackedItemIds.some((itemId) => !resultItemIds.has(itemId)))
    ) {
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
): IngestWizardState => {
  const queueItems = hydrateQueueItems(session.queueItems)
  const pendingRunRequest =
    session.tracking?.submissionState &&
    session.tracking.submissionState !== "creating_run"
      ? buildPlaylistIngestRunRequest(queueItems).request
      : null
  return {
    currentStep: session.currentStep,
    highestStep: Math.max(session.currentStep, 1) as IngestWizardState["highestStep"],
    queueItems,
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
    pendingRunRequest,
    isMinimized:
      session.visibility === "hidden" && session.lifecycle === "processing",
  }
}

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
  const lifecycle = deriveLifecycleFromWizardState(state, session)
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

const buildReviewHandoffRevision = (state: IngestWizardState): string =>
  JSON.stringify({
    currentStep: state.currentStep,
    highestStep: state.highestStep,
    queueItems: buildPersistedQueueItems(state.queueItems),
    pendingRunRequest: state.pendingRunRequest,
    processingBlock: state.processingBlock,
    processingState: state.processingState,
    results: state.results,
  })

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
    jobId: number | null
    status: string
    result?: unknown
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
    const resultRecord =
      job.result !== null && typeof job.result === "object"
        ? (job.result as Record<string, unknown>)
        : null
    const runOutcome = String(resultRecord?.outcome || "").trim().toLowerCase()
    const isDuplicate =
      resultStatus === "ok" &&
      (completedIngestJobIndicatesSkipped(job.result) ||
        runOutcome === "included_existing" ||
        runOutcome === "skipped_existing")
    const resultTitle = resultRecord?.title
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
      collectionItemId: tracking?.jobIdToCollectionItemId?.[String(job.jobId)] ?? null,
      retryAttempt: null,
      idempotencyKey: null,
      title: typeof resultTitle === "string" ? resultTitle : null,
      data: job.result,
      message: isDuplicate ? DUPLICATE_SKIP_MESSAGE : undefined,
    }
  })

const buildProgressFromReattachedJobs = (
  items: WizardQueueItem[],
  jobs: ReattachedQuickIngestJob[],
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
  conferenceBatchMetadata: ConferenceBatchMetadata | null,
  pendingRunRequest: IngestWizardState["pendingRunRequest"],
  options: QuickIngestRequestPayload["common"] & {
    storeRemote: boolean
    reviewBeforeStorage: boolean
    advancedValues?: Record<string, unknown>
    typeDefaults: TypeDefaults
  }
): Promise<QuickIngestRequestPayload> => {
  const validItems = items.filter(
    (item) => item.validation.valid && item.conferenceOverride?.selected !== false
  )
  const entries = validItems
    .filter(
      (item): item is WizardQueueItem & { url: string } =>
        Boolean(item.url) &&
        !(
          pendingRunRequest &&
          item.sourceRef?.kind === "materialized_playlist_item"
        )
    )
    .map((item) => ({
      id: item.id,
      url: item.url,
      type: mapDetectedTypeToEntryType(item.detectedType),
      defaults: buildDefaultsForQueueItem(item, options.typeDefaults),
      playlist: item.playlist,
      conferenceOverride: item.conferenceOverride,
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
        conferenceOverride: item.conferenceOverride,
      }))
  )
  const conferenceItemMetadata = Object.fromEntries(
    validItems.flatMap((item) =>
      item.conferenceOverride
        ? [
            [
              item.id,
              {
                playlist: item.playlist,
                conferenceOverride: item.conferenceOverride,
              },
            ],
          ]
        : []
    )
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
      chunking_mode: options.chunking_mode,
      auto_chunking_goal: options.auto_chunking_goal,
      auto_chunking_use_llm: options.auto_chunking_use_llm,
    },
    advancedValues: options.advancedValues ?? {},
    fileDefaults: options.typeDefaults,
    conferenceBatchMetadata,
    conferenceItemMetadata:
      Object.keys(conferenceItemMetadata).length > 0
        ? conferenceItemMetadata
        : undefined,
    pendingRunRequest,
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
  commitReviewHandoff: (state: IngestWizardState) => boolean
  markInterrupted: (reason?: string) => void
  showSession: () => void
  replaceWithNewDraft: () => QuickIngestSessionRecord
  shouldAttemptPersistedReattach: boolean
  cancellationRequestNonce: number
}

const WizardModalContent: React.FC<WizardModalContentProps> = ({
  open,
  onClose,
  autoProcessQueued = false,
  session,
  markProcessingTracking,
  commitReviewHandoff,
  markInterrupted,
  showSession,
  replaceWithNewDraft,
  shouldAttemptPersistedReattach,
  cancellationRequestNonce,
}) => {
  const { t } = useTranslation(["option"])
  const {
    state,
    minimize,
    restore,
    skipToProcessing,
    updateItemProgress,
    updateProcessingState,
    setResults,
    goToStep,
    applyPlaylistReviewRequired,
    goNext,
  } = useIngestWizard()
  const { currentStep, queueItems, processingState, presetConfig, results } = state
  const [analysisProviderWarning, setAnalysisProviderWarning] = useState<
    string | null
  >(null)
  const connectionState = useConnectionStore((store) => store.state)
  const checkConnection = useConnectionStore((store) => store.checkOnce)
  const activeSessionIdRef = useRef<string | null>(null)
  const resultsRef = useRef(results)
  const hasStartedRunRef = useRef(false)
  const runStartedAtRef = useRef<number | null>(null)
  const cancelledSessionIdsRef = useRef<Set<string>>(new Set())
  const cancelRequestedRef = useRef(false)
  const [replayRequestNonce, setReplayRequestNonce] = useState(0)
  const validQueueItems = useMemo(
    () =>
      queueItems.filter(
        (item) => item.validation.valid && item.conferenceOverride?.selected !== false
      ),
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
  const restoredCreatingRunRef = useRef(
    session.lifecycle === "processing" &&
      session.tracking?.submissionState === "creating_run"
  )
  const persistedReattachTimerRef = useRef<number | null>(null)
  const activeReattachSignatureRef = useRef("")
  const [runSubmissionInFlight, setRunSubmissionInFlight] = useState(false)
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

  useEffect(() => {
    if (!restoredCreatingRunRef.current) return
    hasStartedRunRef.current = true
    updateProcessingState({
      status: "error",
      estimatedRemaining: 0,
    })
    markInterrupted(AMBIGUOUS_RUN_CREATE_RECOVERY_MESSAGE)
  }, [markInterrupted, updateProcessingState])
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
      processing: "Processing content... This may take a few minutes for large files.",
      analyzing: "Processing and indexing content",
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
    if (runSubmissionInFlight && persistedTracking.runId) return
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
    runSubmissionInFlight,
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

  const returnToReview = useCallback(
    (reviewRequired: PlaylistReviewRequiredRecoveryItem[]) => {
      const reviewState = applyPlaylistReviewRequiredState(state, reviewRequired)
      if (!commitReviewHandoff(reviewState)) return
      hasStartedRunRef.current = false
      activeSessionIdRef.current = null
      persistedTrackingRef.current = undefined
      setRunSubmissionInFlight(false)
      applyPlaylistReviewRequired(reviewRequired)
    },
    [applyPlaylistReviewRequired, commitReviewHandoff, state]
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
      const trackedEligibleItems = trackedQueueItems.filter(
        (item) => item.validation.valid && item.conferenceOverride?.selected !== false
      )
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
        outcome
      )
      finalizeRun(outcome === "cancelled" ? "cancelled" : "error", fallbackResults)
    },
    [finalizeRun, trackedQueueItems, validQueueItems]
  )

  const handleCancelAll = useCallback(() => {
    if (cancelRequestedRef.current) return
    cancelRequestedRef.current = true

    if (persistedReattachTimerRef.current != null) {
      window.clearTimeout(persistedReattachTimerRef.current)
      persistedReattachTimerRef.current = null
    }

    const persistedTracking = persistedTrackingRef.current
    const sessionId = String(
      activeSessionIdRef.current || persistedTracking?.sessionId || ""
    ).trim()
    if (sessionId) {
      cancelledSessionIdsRef.current.add(sessionId)
      cancelQuickIngestSessionBestEffort({
        sessionId,
        batchIds: resolveTrackingBatchIds(persistedTracking),
        reason: "user_cancelled",
      })
    }

    finalizeFailure("Cancelled by user.", "cancelled")
  }, [finalizeFailure])

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
      if (message.type === "tldw:quick-ingest/progress") {
        const rawResult = message.payload?.result
        const resultId = String(
          rawResult?.id || message.payload?.occurrenceId || ""
        ).trim()
        const status = mapReattachedJobStatusToProgress(
          String(message.payload?.status || rawResult?.status || "processing"),
          rawResult?.data
        )
        if (
          resultId &&
          status !== "complete" &&
          status !== "failed" &&
          status !== "cancelled"
        ) {
          updateItemProgress({
            id: resultId,
            status,
            progressPercent: status === "queued" ? 0 : 50,
            currentStage: String(message.payload?.status || "Processing"),
            estimatedRemaining: 0,
          })
          return
        }
        const result = normalizeWizardResult(rawResult)
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

      if (message.type === "tldw:quick-ingest/review-required") {
        const reviewRequired = Array.isArray(message.payload?.reviewRequired)
          ? message.payload.reviewRequired
          : []
        if (reviewRequired.length > 0) {
          returnToReview(reviewRequired)
          return
        }
        finalizeFailure(
          "Quick ingest requires review, but no recovery details were returned.",
          "failed"
        )
        return
      }

      if (message.type === "tldw:quick-ingest/failed") {
        const normalizedResults = (message.payload?.results || [])
          .map((item) => normalizeWizardResult(item))
          .filter((item): item is WizardResultItem => Boolean(item))
        if (normalizedResults.length > 0) applyResults(normalizedResults)
        finalizeFailure(
          String(message.payload?.error || "Quick ingest failed."),
          "failed"
        )
        return
      }

      if (message.type === "tldw:quick-ingest/cancelled") {
        const normalizedResults = (message.payload?.results || [])
          .map((item) => normalizeWizardResult(item))
          .filter((item): item is WizardResultItem => Boolean(item))
        if (normalizedResults.length > 0) applyResults(normalizedResults)
        finalizeFailure(
          String(message.payload?.reason || "Cancelled by user."),
          "cancelled"
        )
        return
      }

      if (message.type === "tldw:quick-ingest/interrupted") {
        const error = String(
          message.payload?.error || "Quick ingest recovery was interrupted."
        )
        updateProcessingState({ status: "error", estimatedRemaining: 0 })
        markInterrupted(error)
      }
    },
    [
      applyResults,
      finalizeFailure,
      finalizeRun,
      markInterrupted,
      returnToReview,
      updateItemProgress,
      updateProcessingState,
    ]
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

  const runtimeMessageHandlerRef = useRef(handleRuntimeMessage)
  const markInterruptedRef = useRef(markInterrupted)
  useEffect(() => {
    runtimeMessageHandlerRef.current = handleRuntimeMessage
    markInterruptedRef.current = markInterrupted
  }, [handleRuntimeMessage, markInterrupted])

  const replaySessionId = String(session.tracking?.sessionId || "").trim()
  const replayTrackingMode = session.tracking?.mode
  const replayEligibleLifecycle =
    session.lifecycle === "processing" || session.lifecycle === "interrupted"
  useEffect(() => {
    if (
      !open ||
      !replayEligibleLifecycle ||
      replayTrackingMode !== "extension-runtime" ||
      !replaySessionId
    ) {
      return
    }
    let cancelled = false
    const delays = [0, 250, 750]
    void (async () => {
      let lastError = "Extension recovery is temporarily unavailable."
      for (const delay of delays) {
        if (delay > 0) {
          await new Promise((resolve) => setTimeout(resolve, delay))
        }
        if (cancelled) return
        const response = await queryQuickIngestSession(replaySessionId)
        if (cancelled) return
        if (response?.ok) {
          if (response.event) {
            runtimeMessageHandlerRef.current(
              response.event as QuickIngestRuntimeMessage
            )
          }
          return
        }
        lastError = String(response?.error || lastError)
      }
      markInterruptedRef.current(
        `${lastError} Reopen Quick Ingest to try recovery again.`
      )
    })()
    return () => {
      cancelled = true
    }
  }, [
    open,
    replayEligibleLifecycle,
    replayRequestNonce,
    replaySessionId,
    replayTrackingMode,
  ])

  const startRun = useCallback(async () => {
    if (
      restoredCreatingRunRef.current ||
      hasStartedRunRef.current ||
      validQueueItems.length === 0
    ) {
      return
    }
    hasStartedRunRef.current = true

    try {
      try {
        await tldwClient.initialize()
      } catch {
        // Best effort; background proxy handles auth for direct runtimes.
      }
      if (cancelRequestedRef.current) return

      const requestPayload = await buildQuickIngestPayload(
        validQueueItems,
        state.conferenceBatchMetadata,
        state.pendingRunRequest,
        {
          ...presetConfig.common,
          storeRemote: presetConfig.storeRemote,
          reviewBeforeStorage: presetConfig.reviewBeforeStorage,
          advancedValues: presetConfig.advancedValues,
          typeDefaults: presetConfig.typeDefaults,
        }
      )
      if (cancelRequestedRef.current) return

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

      const startAck = await startQuickIngestSession(requestPayload)
      if (cancelRequestedRef.current) {
        const cancelledSessionId = String(startAck?.sessionId || "").trim()
        if (startAck?.ok && cancelledSessionId) {
          cancelledSessionIdsRef.current.add(cancelledSessionId)
          if (!cancelledSessionId.startsWith("qi-direct-")) {
            cancelQuickIngestSessionBestEffort({
              sessionId: cancelledSessionId,
              reason: "user_cancelled",
            })
          }
        }
        return
      }
      if (!startAck?.ok || !startAck?.sessionId) {
        const indeterminateSessionId = String(startAck?.sessionId || "").trim()
        if (startAck?.indeterminate && indeterminateSessionId) {
          activeSessionIdRef.current = indeterminateSessionId
          markProcessingTracking({
            mode: "extension-runtime",
            sessionId: indeterminateSessionId,
            ...(requestPayload.pendingRunRequest
              ? {
                  submissionOccurrenceIds:
                    requestPayload.pendingRunRequest.inputs.map(
                      (input) => input.occurrenceId
                    ),
                }
              : {}),
            startedAt: runStartedAtRef.current || Date.now(),
          })
          updateProcessingState({ status: "error", estimatedRemaining: 0 })
          markInterrupted(
            startAck.error ||
              "Quick ingest start was accepted but its response was interrupted."
          )
          return
        }
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
      const extensionRuntime = !sessionId.startsWith("qi-direct-")
      if (extensionRuntime || !requestPayload.pendingRunRequest) {
        markProcessingTracking({
          mode: extensionRuntime ? "extension-runtime" : "webui-direct",
          sessionId,
          ...(extensionRuntime && requestPayload.pendingRunRequest
            ? {
                submissionOccurrenceIds:
                  requestPayload.pendingRunRequest.inputs.map(
                    (input) => input.occurrenceId
                  ),
              }
            : {}),
          startedAt: runStartedAtRef.current || Date.now(),
        })
      }

      if (!sessionId.startsWith("qi-direct-")) {
        return
      }

      setRunSubmissionInFlight(Boolean(requestPayload.pendingRunRequest))
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
      if (!response?.submissionCleanupFailed) {
        setRunSubmissionInFlight(false)
      }

      if (
        cancelledSessionIdsRef.current.has(sessionId) ||
        sessionId !== String(activeSessionIdRef.current || "").trim()
      ) {
        return
      }

      if (response?.reviewRequired?.length) {
        returnToReview(response.reviewRequired)
        return
      }

      if (response?.submissionCleanupFailed) {
        if (response.accepted) {
          const unsentOccurrenceIds = new Set(response.unsentOccurrenceIds || [])
          applyResults(
            buildFailureResults(
              validQueueItems.filter((item) => unsentOccurrenceIds.has(item.id)),
              response.error || "Playlist ingest submission stopped. Try again.",
              "failed"
            )
          )
        }
        const cleanupError =
          response.error ||
          "The server could not cancel unsent items. Retry cancellation before reconnecting."
        updateProcessingState({
          status: "error",
          estimatedRemaining: 0,
        })
        hasStartedRunRef.current = false
        activeSessionIdRef.current = null
        markInterrupted(cleanupError)
        return
      }

      if (response?.submissionBlocked && response?.accepted) {
        const unsentOccurrenceIds = new Set(response.unsentOccurrenceIds || [])
        applyResults(
          buildFailureResults(
            validQueueItems.filter((item) => unsentOccurrenceIds.has(item.id)),
            response.error || "Playlist ingest submission stopped. Try again.",
            "failed"
          )
        )
        return
      }

      if (response?.runId && response?.accepted) {
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
      setRunSubmissionInFlight(false)
      if (cancelRequestedRef.current) return
      finalizeFailure(
        error instanceof Error ? error.message : "Quick ingest failed.",
        "failed"
      )
    }
  }, [
    applyResults,
    finalizeFailure,
    finalizeRun,
    markRunActive,
    markInterrupted,
    presetConfig.advancedValues,
    presetConfig.common,
    presetConfig.reviewBeforeStorage,
    presetConfig.storeRemote,
    presetConfig.typeDefaults,
    state.conferenceBatchMetadata,
    state.pendingRunRequest,
    goToStep,
    restore,
    showSession,
    markProcessingTracking,
    qi,
    returnToReview,
    updateProcessingState,
    validQueueItems,
  ])

  useEffect(() => {
    if (currentStep !== 4 || processingState.status !== "running") return
    if (session.tracking?.mode === "extension-runtime") {
      return
    }
    if (session.lifecycle === "processing" && session.tracking) {
      const canReattachDirectJobs =
        session.tracking.mode === "webui-direct" &&
        Boolean(session.tracking.jobIds?.length || session.tracking.runId)
      if (session.tracking.mode !== "webui-direct" || canReattachDirectJobs) {
        return
      }
    }
    void startRun()
  }, [currentStep, processingState.status, session.lifecycle, session.tracking, startRun])

  useEffect(() => {
    if (cancellationRequestNonce <= 0) return
    const persistedTracking = persistedTrackingRef.current
    const sessionId = String(
      activeSessionIdRef.current || persistedTracking?.sessionId || ""
    ).trim()
    if (!sessionId || cancelledSessionIdsRef.current.has(sessionId)) return
    cancelledSessionIdsRef.current.add(sessionId)
    for (const progress of processingState.perItemProgress) {
      if (
        progress.status === "complete" ||
        progress.status === "failed" ||
        progress.status === "cancelled"
      ) {
        continue
      }
      updateItemProgress({
        ...progress,
        status: "processing",
        currentStage: "Cancellation requested",
        estimatedRemaining: 0,
      })
    }
    void cancelQuickIngestSession({
      sessionId,
      batchIds: resolveTrackingBatchIds(persistedTracking),
      tracking: persistedTracking,
      reason: "user_cancelled",
    })
      .then((response) => {
        if (response.ok) return
        cancelledSessionIdsRef.current.delete(sessionId)
        updateProcessingState({ status: "error", estimatedRemaining: 0 })
        markInterrupted(
          response.error ||
            "Quick ingest cancellation was not confirmed. Recovery will reconcile the run."
        )
        setReplayRequestNonce((value) => value + 1)
      })
      .catch((error) => {
        cancelledSessionIdsRef.current.delete(sessionId)
        updateProcessingState({ status: "error", estimatedRemaining: 0 })
        markInterrupted(
          error instanceof Error
            ? error.message
            : "Quick ingest cancellation was not confirmed. Recovery will reconcile the run."
        )
        setReplayRequestNonce((value) => value + 1)
      })
  }, [
    cancellationRequestNonce,
    processingState.perItemProgress,
    markInterrupted,
    updateItemProgress,
    updateProcessingState,
  ])

  useEffect(() => {
    if (processingState.status !== "cancelled") return
    const persistedTracking = persistedTrackingRef.current
    const sessionId = String(
      activeSessionIdRef.current || persistedTracking?.sessionId || ""
    ).trim()
    if (!sessionId || cancelledSessionIdsRef.current.has(sessionId)) return
    cancelledSessionIdsRef.current.add(sessionId)
    const waitsForRuntimeTerminal =
      persistedTracking?.mode === "extension-runtime" ||
      Boolean(persistedTracking?.runId)
    void cancelQuickIngestSession({
      sessionId,
      batchIds: resolveTrackingBatchIds(persistedTracking),
      tracking: persistedTracking,
      reason: "user_cancelled",
    })
      .then((response) => {
        if (!response.ok || waitsForRuntimeTerminal) return
        finalizeFailure("Cancelled by user.", "cancelled")
      })
      .catch(() => {
        if (!waitsForRuntimeTerminal) {
          finalizeFailure("Cancelled by user.", "cancelled")
        }
      })
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
  }, [handleCancelAll, isProcessingActive, qi, minimize, onClose])

  // Quick-process callback for AddContentStep (skip to processing with defaults)
  const handleQuickProcess = useCallback(() => {
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
      if (currentStep === 1) {
        goNext()
      } else {
        goToStep(2)
      }
      return
    }
    setAnalysisProviderWarning(null)
    skipToProcessing()
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
      handleQuickProcess()
    }
  }, [
    autoProcessQueued,
    handleQuickProcess,
    isCheckingConnection,
    isOnlineForIngest,
    validQueueItems.length,
  ])

  // Navigation callbacks for WizardResultsStep CTAs
  const navigate = useNavigate()

  const handleSearchKnowledge = useCallback(() => {
    navigate("/knowledge")
    onClose()
  }, [navigate, onClose])

  const handleIngestMore = useCallback(() => {
    replaceWithNewDraft()
  }, [replaceWithNewDraft])

  const handleOpenWorkspace = useCallback(
    (item: WizardResultItem) => {
      const mediaId = item.mediaId
      if (mediaId != null) {
        navigate(`${DOCUMENT_WORKSPACE_PATH}?open=${mediaId}`)
      } else {
        navigate(DOCUMENT_WORKSPACE_PATH)
      }
      onClose()
    },
    [navigate, onClose]
  )

  const handleOpenMedia = useCallback(
    (item: WizardResultItem) => {
      const mediaId = item.mediaId
      const mediaPath = mediaId != null
        ? `/media?id=${encodeURIComponent(String(mediaId))}`
        : "/media"
      navigate(mediaPath)
      onClose()
    },
    [navigate, onClose]
  )

  const handleOpenCollection = useCallback(
    (collectionId: string) => {
      const collectionPath = buildMediaCollectionReviewPath(collectionId)
      onClose()
      navigate(collectionPath)
    },
    [navigate, onClose]
  )

  // Render the current step
  const stepContent = useMemo(() => {
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
            isStepVisible={open && !state.isMinimized && currentStep === 2}
            analysisProviderWarning={analysisProviderWarning}
            focusAnalysisProvider={Boolean(analysisProviderWarning)}
          />
        )
      case 3:
        return (
          <ReviewStep
            isOnlineForIngest={isOnlineForIngest}
            isCheckingConnection={isCheckingConnection}
            connectionRecoveryMessage={connectionRecoveryMessage}
            onRetryConnection={handleRetryConnection}
          />
        )
      case 4:
        return <ProcessingStep onCancelAll={handleCancelAll} />
      case 5:
        return (
          <WizardResultsStep
            onClose={onClose}
            onIngestMore={handleIngestMore}
            onOpenMedia={handleOpenMedia}
            onSearchKnowledge={handleSearchKnowledge}
            onOpenWorkspace={handleOpenWorkspace}
            onOpenCollection={handleOpenCollection}
          />
        )
      default:
        return null
    }
  }, [
    connectionRecoveryMessage,
    currentStep,
    handleCancelAll,
    handleOpenMedia,
    handleOpenCollection,
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
  ])

  return (
    <>
      <Modal
        open={open && !state.isMinimized}
        onCancel={handleCloseAttempt}
        title={modalTitle}
        footer={null}
        width={800}
        className="quick-ingest-modal quick-ingest-wizard-modal"
        getContainer={false}
        styles={QUICK_INGEST_MODAL_STYLES}
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
  presetMap = DEFAULT_PRESETS,
  openRevision = 0,
  createNewDraft,
}) => {
  const {
    session,
    upsertSession,
    markProcessingTracking,
    commitReviewHandoff: commitReviewHandoffInStore,
    markInterrupted: markInterruptedInStore,
    createDraftSession,
    showSession,
    replaceWithNewDraft,
  } =
    useQuickIngestSessionStore(
      useShallow((store) => ({
        session: store.session,
        upsertSession: store.upsertSession,
        markProcessingTracking: store.markProcessingTracking,
        commitReviewHandoff: store.commitReviewHandoff,
        markInterrupted: store.markInterrupted,
        createDraftSession: store.createDraftSession,
        showSession: store.showSession,
        replaceWithNewDraft: store.replaceWithNewDraft,
      }))
    )

  const initialState = useMemo(
    () => (session ? buildInitialWizardState(session) : undefined),
    [session]
  )
  const sessionRef = useRef(session)
  const lastPersistedSignatureRef = useRef<{
    sessionId: string
    signature: string
  } | null>(null)
  const reviewHandoffGuardRef = useRef<string | null>(null)
  const [cancellationRequestNonce, setCancellationRequestNonce] = useState(0)

  useEffect(() => {
    sessionRef.current = session
  }, [session])

  const markSessionInterrupted = useCallback(
    (reason?: string) => {
      if (sessionRef.current) {
        sessionRef.current = {
          ...sessionRef.current,
          lifecycle: "interrupted",
          errorMessage: reason || "Quick ingest was interrupted.",
        }
      }
      markInterruptedInStore(reason)
    },
    [markInterruptedInStore]
  )

  useEffect(() => {
    if (!open || session) return
    createDraftSession({
      selectedPreset: DEFAULT_PRESET,
      customBasePreset: DEFAULT_PRESET,
      presetConfig: presetMap[DEFAULT_PRESET],
      customOptions: {},
    })
  }, [createDraftSession, open, presetMap, session])

  const persistWizardState = useCallback(
    (state: IngestWizardState) => {
      const handoffRevision = reviewHandoffGuardRef.current
      if (handoffRevision) {
        if (buildReviewHandoffRevision(state) !== handoffRevision) return
        reviewHandoffGuardRef.current = null
        return
      }
      const currentSession = sessionRef.current
      if (!currentSession) return
      const patch = buildSessionPatchFromWizardState(state, currentSession)
      if (patch.completedAt == null) {
        lastPersistedSignatureRef.current = null
        upsertSession(patch)
        return
      }
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
    [upsertSession]
  )

  const commitSessionReviewHandoff = useCallback(
    (state: IngestWizardState): boolean => {
      const currentSession = sessionRef.current
      if (!currentSession) return false
      const revision = buildReviewHandoffRevision(state)
      reviewHandoffGuardRef.current = revision
      let committed = false
      try {
        committed = commitReviewHandoffInStore(
          buildSessionPatchFromWizardState(state, currentSession)
        )
      } catch {
        reviewHandoffGuardRef.current = null
        return false
      }
      if (!committed) {
        reviewHandoffGuardRef.current = null
        return false
      }
      sessionRef.current = useQuickIngestSessionStore.getState().session
      return true
    },
    [commitReviewHandoffInStore]
  )

  const deferAuthoritativeCancellation = useCallback(() => {
    const tracking = sessionRef.current?.tracking
    if (tracking?.mode !== "extension-runtime" && !tracking?.runId) {
      return false
    }
    setCancellationRequestNonce((value) => value + 1)
    return true
  }, [])

  if (!session || !initialState) return null

  const providerKey = `${session.id}:${openRevision}`

  return (
    <IngestWizardProvider
      key={providerKey}
      initialState={initialState}
      onStateChange={persistWizardState}
      presetMap={presetMap}
      onCancelProcessing={deferAuthoritativeCancellation}
    >
      <WizardModalContent
        open={open}
        onClose={onClose}
        autoProcessQueued={autoProcessQueued}
        session={session}
        markProcessingTracking={markProcessingTracking}
        commitReviewHandoff={commitSessionReviewHandoff}
        markInterrupted={markSessionInterrupted}
        showSession={showSession}
        replaceWithNewDraft={createNewDraft ?? replaceWithNewDraft}
        cancellationRequestNonce={cancellationRequestNonce}
        shouldAttemptPersistedReattach={
          session.lifecycle === "processing" &&
          session.tracking?.mode === "webui-direct" &&
          Boolean(session.tracking?.jobIds?.length || session.tracking?.runId)
        }
      />
    </IngestWizardProvider>
  )
}

export default QuickIngestWizardModal
