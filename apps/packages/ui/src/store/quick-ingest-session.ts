import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

import {
  createQuickIngestIndexedDbStorage,
  type QuickIngestIndexedDbStorage,
  type QuickIngestPersistenceStatus,
} from "@/db/dexie/quick-ingest"

import {
  playlistHasMaterializationCues,
  type IngestPreset,
  type PresetConfig,
  type QueueItemValidation,
  type WizardProcessingState,
  type WizardResultItem,
  type WizardStep,
  type DetectedMediaType,
  type ConferenceBatchMetadata,
  type ConferenceItemMetadataOverride,
  type PlaylistQueueMetadata,
  type PlaylistReviewState,
  type WizardSourceRef,
} from "@/components/Common/QuickIngest/types"
import { DEFAULT_PRESET, DEFAULT_PRESETS } from "@/components/Common/QuickIngest/presets"
import {
  isFirstSourceQuickIngestKind,
  type FirstSourceQuickIngestKind,
  type QuickIngestOpenDetail,
} from "@/utils/quick-ingest-open"

const STORAGE_KEY = "tldw-quick-ingest-session"

export type QuickIngestSessionLifecycle =
  | "draft"
  | "processing"
  | "completed"
  | "partial_failure"
  | "cancelled"
  | "interrupted"

export type PersistedQuickIngestTracking = {
  mode: "webui-direct" | "extension-runtime" | "unknown"
  submissionState?:
    | "creating_run"
    | "run_created"
    | "submitting"
    | "cleanup_required"
    | "acknowledged"
  submissionOccurrenceIds?: string[]
  sessionId?: string
  runId?: string
  generation?: string
  batchId?: string
  batchIds?: string[]
  collectionId?: string
  plannedItemIds?: string[]
  jobIds?: number[]
  submittedItemIds?: string[]
  /** @deprecated use submittedItemIds */
  itemIds?: string[]
  jobIdToItemId?: Record<string, string>
  jobIdToCollectionItemId?: Record<string, string>
  durableMode?: "durable_collection" | "degraded" | "unknown"
  startedAt?: number
}

export type PersistedWizardQueueItem = {
  id: string
  sourceRef?: WizardSourceRef
  kind?: string
  fileName?: string
  name?: string
  key?: string
  size?: number
  type?: string
  lastModified?: number
  url?: string
  detectedType: DetectedMediaType
  icon: string
  fileSize: number
  mimeType?: string
  validation: QueueItemValidation
  playlist?: PlaylistQueueMetadata
  playlistReview?: PlaylistReviewState
  conferenceOverride?: ConferenceItemMetadataOverride
  fileStub?: {
    key?: string
    instanceId?: string
    lastModified?: number
  }
}

export type QuickIngestTriggerSummary = {
  count: number
  label: string | null
  hadFailure: boolean
}

export type QuickIngestSessionBadge = {
  queueCount: number
  hasRecentFailure: boolean
}

export type QuickIngestSessionResultSummary = {
  status: "idle" | "success" | "error" | "cancelled"
  attemptedAt: number | null
  completedAt: number | null
  totalCount: number
  successCount: number
  failedCount: number
  cancelledCount: number
  firstMediaId: string | null
  primarySourceLabel: string | null
  errorMessage: string | null
}

export type QuickIngestSessionRecord = {
  id: string
  visibility: "visible" | "hidden"
  lifecycle: QuickIngestSessionLifecycle
  currentStep: WizardStep
  queueItems: PersistedWizardQueueItem[]
  selectedPreset: IngestPreset
  customBasePreset: Exclude<IngestPreset, "custom">
  presetConfig: PresetConfig
  customOptions: Partial<PresetConfig>
  processingState: WizardProcessingState
  results: WizardResultItem[]
  openDetail?: QuickIngestOpenDetail | null
  firstSourceAddMode?: FirstSourceQuickIngestKind | null
  conferenceBatchMetadata?: ConferenceBatchMetadata | null
  badge: QuickIngestSessionBadge
  resultSummary: QuickIngestSessionResultSummary
  tracking?: PersistedQuickIngestTracking
  errorMessage?: string | null
  createdAt: number
  updatedAt: number
  completedAt?: number | null
}

type QuickIngestSessionPersistedState = {
  session: QuickIngestSessionRecord | null
}

const isCustomBasePreset = (
  value: unknown
): value is Exclude<IngestPreset, "custom"> =>
  typeof value === "string" && value in DEFAULT_PRESETS

type QuickIngestSessionState = QuickIngestSessionPersistedState & {
  triggerSummary: QuickIngestTriggerSummary
  persistenceStatus: QuickIngestPersistenceStatus
  isSubmissionOwner: boolean
  externalAuthorityRevision: number
  createDraftSession: (
    seed?: Partial<QuickIngestSessionRecord>
  ) => QuickIngestSessionRecord
  upsertSession: (next: Partial<QuickIngestSessionRecord>) => void
  showSession: () => void
  hideSession: () => void
  markProcessingTracking: (tracking: PersistedQuickIngestTracking) => void
  clearProcessingTracking: () => void
  commitReviewHandoff: (
    next: Partial<QuickIngestSessionRecord>
  ) => Promise<boolean>
  commitProcessingHandoff: (
    next: Partial<QuickIngestSessionRecord>,
    tracking: PersistedQuickIngestTracking
  ) => Promise<boolean>
  acquireSubmissionLease: () => Promise<boolean>
  renewSubmissionLease: () => Promise<boolean>
  releaseSubmissionLease: () => Promise<void>
  markInterrupted: (reason?: string) => void
  clearSession: () => void
  replaceWithNewDraft: (
    seed?: Partial<QuickIngestSessionRecord>
  ) => QuickIngestSessionRecord
}

const INITIAL_PROCESSING_STATE: WizardProcessingState = {
  status: "idle",
  perItemProgress: [],
  elapsed: 0,
  estimatedRemaining: 0,
}

const INITIAL_RESULT_SUMMARY: QuickIngestSessionResultSummary = {
  status: "idle",
  attemptedAt: null,
  completedAt: null,
  totalCount: 0,
  successCount: 0,
  failedCount: 0,
  cancelledCount: 0,
  firstMediaId: null,
  primarySourceLabel: null,
  errorMessage: null,
}

const generateSessionId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `qi-session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

const normalizeStringIds = (values?: unknown[]): string[] =>
  Array.from(
    new Set(
      Array.isArray(values)
        ? values
            .map((value) => String(value || "").trim())
            .filter(Boolean)
        : []
    )
  )

const MAX_PERSISTED_RUN_ID_LENGTH = 255
const MAX_PERSISTED_SUBMISSION_OCCURRENCES = 500

const boundedTrackingIds = (values: unknown[]): string[] =>
  normalizeStringIds(values)
    .filter((value) => value.length <= MAX_PERSISTED_RUN_ID_LENGTH)
    .slice(0, MAX_PERSISTED_SUBMISSION_OCCURRENCES)

const sanitizeRunId = (value?: string): string | undefined => {
  const runId = value?.trim() || ""
  return runId.length > 0 && runId.length <= MAX_PERSISTED_RUN_ID_LENGTH
    ? runId
    : undefined
}

const sanitizeTracking = (
  tracking?: PersistedQuickIngestTracking
): PersistedQuickIngestTracking | undefined => {
  if (!tracking) return undefined
  const batchIds = boundedTrackingIds([
    ...(Array.isArray(tracking.batchIds) ? tracking.batchIds : []),
    tracking.batchId,
  ])
  const jobIds = Array.isArray(tracking.jobIds)
    ? Array.from(
        new Set(
          tracking.jobIds
            .map((jobId) => Number(jobId))
            .filter((jobId) => Number.isFinite(jobId) && jobId > 0)
            .map((jobId) => Math.trunc(jobId))
            .filter((jobId) => Number.isSafeInteger(jobId))
        )
      ).slice(0, MAX_PERSISTED_SUBMISSION_OCCURRENCES)
    : undefined
  const submittedItemIds = boundedTrackingIds([
    ...(Array.isArray(tracking.submittedItemIds)
      ? tracking.submittedItemIds
      : []),
    ...(Array.isArray(tracking.itemIds) ? tracking.itemIds : []),
  ])
  const plannedItemIds = boundedTrackingIds(
    Array.isArray(tracking.plannedItemIds) ? tracking.plannedItemIds : []
  )
  const submissionOccurrenceIds = boundedTrackingIds(
    Array.isArray(tracking.submissionOccurrenceIds)
      ? tracking.submissionOccurrenceIds
      : []
  )
  const jobIdToItemIdEntries = Object.entries(tracking.jobIdToItemId || {})
    .map(([jobId, itemId]) => [String(jobId || "").trim(), String(itemId || "").trim()] as const)
    .filter(
      ([jobId, itemId]) =>
        jobId.length > 0 &&
        jobId.length <= MAX_PERSISTED_RUN_ID_LENGTH &&
        itemId.length > 0 &&
        itemId.length <= MAX_PERSISTED_RUN_ID_LENGTH
    )
    .slice(0, MAX_PERSISTED_SUBMISSION_OCCURRENCES)
  const jobIdToCollectionItemIdEntries = Object.entries(
    tracking.jobIdToCollectionItemId || {}
  )
    .map(([jobId, itemId]) => [String(jobId || "").trim(), String(itemId || "").trim()] as const)
    .filter(
      ([jobId, itemId]) =>
        jobId.length > 0 &&
        jobId.length <= MAX_PERSISTED_RUN_ID_LENGTH &&
        itemId.length > 0 &&
        itemId.length <= MAX_PERSISTED_RUN_ID_LENGTH
    )
    .slice(0, MAX_PERSISTED_SUBMISSION_OCCURRENCES)
  const normalizedMode =
    tracking.mode === "webui-direct" ||
    tracking.mode === "extension-runtime" ||
    tracking.mode === "unknown"
      ? tracking.mode
      : "unknown"

  return {
    mode: normalizedMode,
    submissionState:
      tracking.submissionState === "creating_run" ||
      tracking.submissionState === "run_created" ||
      tracking.submissionState === "submitting" ||
      tracking.submissionState === "cleanup_required" ||
      tracking.submissionState === "acknowledged"
        ? tracking.submissionState
        : undefined,
    submissionOccurrenceIds:
      submissionOccurrenceIds.length > 0 ? submissionOccurrenceIds : undefined,
    sessionId: sanitizeRunId(tracking.sessionId),
    runId: sanitizeRunId(tracking.runId),
    generation: sanitizeRunId(tracking.generation),
    batchId:
      sanitizeRunId(tracking.batchId) ||
      (batchIds.length > 0 ? batchIds[batchIds.length - 1] : undefined),
    batchIds: batchIds.length > 0 ? batchIds : undefined,
    collectionId: sanitizeRunId(tracking.collectionId),
    plannedItemIds: plannedItemIds.length > 0 ? plannedItemIds : undefined,
    jobIds: jobIds && jobIds.length > 0 ? jobIds : undefined,
    submittedItemIds:
      submittedItemIds.length > 0 ? submittedItemIds : undefined,
    itemIds: submittedItemIds.length > 0 ? submittedItemIds : undefined,
    jobIdToItemId:
      jobIdToItemIdEntries.length > 0
        ? Object.fromEntries(jobIdToItemIdEntries)
        : undefined,
    jobIdToCollectionItemId:
      jobIdToCollectionItemIdEntries.length > 0
        ? Object.fromEntries(jobIdToCollectionItemIdEntries)
        : undefined,
    durableMode:
      tracking.durableMode === "durable_collection" ||
      tracking.durableMode === "degraded" ||
      tracking.durableMode === "unknown"
        ? tracking.durableMode
        : undefined,
    startedAt:
      typeof tracking.startedAt === "number" && Number.isFinite(tracking.startedAt)
        ? tracking.startedAt
        : undefined,
  }
}

const mergeTracking = (
  current?: PersistedQuickIngestTracking,
  incoming?: PersistedQuickIngestTracking
): PersistedQuickIngestTracking | undefined => {
  const base = sanitizeTracking(current)
  const next = sanitizeTracking(incoming)

  if (!base && !next) return undefined
  if (!base) return next
  if (!next) return base
  if (base.sessionId && next.sessionId && base.sessionId !== next.sessionId) {
    return next
  }

  return sanitizeTracking({
    mode: next.mode !== "unknown" ? next.mode : base.mode,
    submissionState: next.submissionState || base.submissionState,
    submissionOccurrenceIds: [
      ...(base.submissionOccurrenceIds || []),
      ...(next.submissionOccurrenceIds || []),
    ],
    sessionId: next.sessionId || base.sessionId,
    runId: next.runId || base.runId,
    generation: next.generation || base.generation,
    batchId: next.batchId || base.batchId,
    batchIds: [...(base.batchIds || []), ...(next.batchIds || [])],
    collectionId: next.collectionId || base.collectionId,
    plannedItemIds: [
      ...(base.plannedItemIds || []),
      ...(next.plannedItemIds || []),
    ],
    jobIds: [...(base.jobIds || []), ...(next.jobIds || [])],
    submittedItemIds: [
      ...(base.submittedItemIds || base.itemIds || []),
      ...(next.submittedItemIds || next.itemIds || []),
    ],
    jobIdToItemId: {
      ...(base.jobIdToItemId || {}),
      ...(next.jobIdToItemId || {}),
    },
    jobIdToCollectionItemId: {
      ...(base.jobIdToCollectionItemId || {}),
      ...(next.jobIdToCollectionItemId || {}),
    },
    durableMode: next.durableMode || base.durableMode,
    startedAt: base.startedAt || next.startedAt,
  })
}

// Defensive ceiling for restored source rows. Overflow is represented by an
// invalid sentinel row below so a truncated draft can never appear complete.
const MAX_PERSISTED_QUEUE_SOURCE_ITEMS = 1000
const PERSISTED_QUEUE_OVERFLOW_ERROR =
  "This draft exceeded the 1000-source persistence safety limit. Start a new batch for the omitted sources."
const MAX_ID_LENGTH = 255
const MAX_DISPLAY_LENGTH = 2000
const MAX_REVIEW_PATCH_LENGTH = 500
const MAX_URL_LENGTH = 8192
const MAX_VALIDATION_MESSAGES = 20
const MAX_KEYWORDS = 100
const MAX_KEYWORD_LENGTH = 128

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const boundedString = (value: unknown, maxLength: number): string | undefined => {
  if (typeof value !== "string" || !value.trim() || value.length > maxLength) return undefined
  return value
}

const canonicalId = (value: unknown): string | undefined => {
  const id = boundedString(value, MAX_ID_LENGTH)
  return id && id.trim() === id ? id : undefined
}

const boundedStringArray = (
  value: unknown,
  maxCount: number,
  maxLength: number
): string[] | undefined => {
  if (!Array.isArray(value) || value.length > maxCount) return undefined
  const strings = value.flatMap((entry) => {
    const next = boundedString(entry, maxLength)
    return next ? [next] : []
  })
  return strings.length > 0 ? strings : undefined
}

const sanitizeSourceRef = (value: unknown, itemId: string): WizardSourceRef | undefined => {
  const sourceRef = asRecord(value)
  if (!sourceRef || canonicalId(sourceRef.occurrenceId) !== itemId) return undefined
  if (sourceRef.kind === "materialized_playlist_item") {
    const materializationId = canonicalId(sourceRef.materializationId)
    return materializationId
      ? { kind: "materialized_playlist_item", materializationId, occurrenceId: itemId }
      : undefined
  }
  if (sourceRef.kind === "direct_url") {
    const url = boundedString(sourceRef.url, MAX_URL_LENGTH)
    return url ? { kind: "direct_url", occurrenceId: itemId, url } : undefined
  }
  return sourceRef.kind === "file_stub"
    ? { kind: "file_stub", occurrenceId: itemId }
    : undefined
}

const sanitizePlaylist = (value: unknown): PlaylistQueueMetadata | undefined => {
  const playlist = asRecord(value)
  if (!playlist) return undefined
  const duplicateStatus =
    playlist.duplicateStatus === "new" ||
    playlist.duplicateStatus === "duplicate_in_batch" ||
    playlist.duplicateStatus === "duplicate_existing" ||
    playlist.duplicateStatus === "unknown"
      ? playlist.duplicateStatus
      : undefined
  const ordinal =
    typeof playlist.ordinal === "number" &&
    Number.isSafeInteger(playlist.ordinal) &&
    playlist.ordinal > 0 &&
    playlist.ordinal <= 1_000_000
      ? playlist.ordinal
      : undefined
  const durationSeconds =
    typeof playlist.durationSeconds === "number" &&
    Number.isFinite(playlist.durationSeconds) &&
    playlist.durationSeconds >= 0
      ? playlist.durationSeconds
      : undefined
  const expiresAt = boundedString(playlist.materializationExpiresAt, 64)
  const materializationExpiresAt =
    expiresAt && Number.isFinite(Date.parse(expiresAt)) ? expiresAt : undefined
  const next: PlaylistQueueMetadata = {
    ...(boundedString(playlist.playlistId, MAX_ID_LENGTH)
      ? { playlistId: playlist.playlistId as string }
      : {}),
    ...(boundedString(playlist.playlistTitle, MAX_DISPLAY_LENGTH)
      ? { playlistTitle: playlist.playlistTitle as string }
      : {}),
    ...(ordinal ? { ordinal } : {}),
    ...(boundedString(playlist.title, MAX_DISPLAY_LENGTH)
      ? { title: playlist.title as string }
      : {}),
    ...(boundedString(playlist.channelOrUploader, MAX_DISPLAY_LENGTH)
      ? { channelOrUploader: playlist.channelOrUploader as string }
      : {}),
    ...(durationSeconds !== undefined ? { durationSeconds } : {}),
    ...(boundedString(playlist.normalizedSourceId, MAX_DISPLAY_LENGTH)
      ? { normalizedSourceId: playlist.normalizedSourceId as string }
      : {}),
    ...(duplicateStatus ? { duplicateStatus } : {}),
    ...(boundedString(playlist.sourceUrl, MAX_URL_LENGTH)
      ? { sourceUrl: playlist.sourceUrl as string }
      : {}),
    ...(materializationExpiresAt ? { materializationExpiresAt } : {}),
  }
  return Object.keys(next).length > 0 ? next : undefined
}

const DUPLICATE_POLICIES = new Set([
  "skip",
  "overwrite",
  "update_metadata_only",
  "include_existing",
])

const sanitizePlaylistReview = (value: unknown): PlaylistReviewState | undefined => {
  const review = asRecord(value)
  if (!review) return undefined
  const duplicatePolicy =
    typeof review.duplicatePolicy === "string" && DUPLICATE_POLICIES.has(review.duplicatePolicy)
      ? (review.duplicatePolicy as PlaylistReviewState["duplicatePolicy"])
      : undefined
  const hasAllowedDuplicatePolicies = Array.isArray(review.allowedDuplicatePolicies)
  const allowedDuplicatePolicies = hasAllowedDuplicatePolicies
    ? Array.from(
        new Set(
          review.allowedDuplicatePolicies.filter(
            (policy): policy is "skip" | "overwrite" | "update_metadata_only" | "include_existing" =>
              typeof policy === "string" && DUPLICATE_POLICIES.has(policy)
          )
        )
      ).slice(0, 4)
    : []
  const evidence = asRecord(review.duplicateEvidence)
  const duplicateEvidence = (() => {
    if (!evidence) return undefined
    if (evidence.kind === "none") {
      return { kind: "none" as const, existingMediaId: null, duplicateOfOccurrenceId: null }
    }
    if (
      evidence.kind === "library" &&
      typeof evidence.existingMediaId === "number" &&
      Number.isSafeInteger(evidence.existingMediaId) &&
      evidence.existingMediaId > 0
    ) {
      return {
        kind: "library" as const,
        existingMediaId: evidence.existingMediaId,
        duplicateOfOccurrenceId: null,
      }
    }
    const duplicateOfOccurrenceId = canonicalId(evidence.duplicateOfOccurrenceId)
    return evidence.kind === "in_run" && duplicateOfOccurrenceId
      ? {
          kind: "in_run" as const,
          existingMediaId: null,
          duplicateOfOccurrenceId,
        }
      : undefined
  })()
  const rawPatch = asRecord(review.metadataPatch)
  const title = boundedString(rawPatch?.title, MAX_REVIEW_PATCH_LENGTH)
  const author = boundedString(rawPatch?.author, MAX_REVIEW_PATCH_LENGTH)
  const rawKeywords = rawPatch?.keywordsAdd
  const keywords = boundedStringArray(rawKeywords, MAX_KEYWORDS, MAX_KEYWORD_LENGTH)
  const keywordKeys = new Set<string>()
  const keywordsAdd = keywords?.filter((keyword) => {
    const key = keyword.toLocaleLowerCase()
    if (keywordKeys.has(key)) return false
    keywordKeys.add(key)
    return true
  })
  const metadataPatch = {
    ...(title ? { title } : {}),
    ...(author ? { author } : {}),
    ...(keywordsAdd ? { keywordsAdd } : {}),
  }
  const editedFieldSet = new Set(
    Array.isArray(review.editedFields)
      ? review.editedFields.filter(
          (field): field is "title" | "author" | "keywordsAdd" =>
            field === "title" || field === "author" || field === "keywordsAdd"
        )
      : []
  )
  const editedFields = (["title", "author", "keywordsAdd"] as const).filter(
    (field) => editedFieldSet.has(field) && field in metadataPatch
  )
  return {
    selected: typeof review.selected === "boolean" ? review.selected : true,
    ...(duplicatePolicy ? { duplicatePolicy } : {}),
    ...(duplicateEvidence ? { duplicateEvidence } : {}),
    ...(hasAllowedDuplicatePolicies ? { allowedDuplicatePolicies } : {}),
    ...(boundedString(review.reviewReason, MAX_DISPLAY_LENGTH)
      ? { reviewReason: review.reviewReason as string }
      : {}),
    ...(Object.keys(metadataPatch).length > 0 ? { metadataPatch } : {}),
    ...(editedFields.length > 0 ? { editedFields } : {}),
  }
}

const sanitizeConferenceOverride = (
  value: unknown
): ConferenceItemMetadataOverride | undefined => {
  const override = asRecord(value)
  if (!override) return undefined
  const duplicatePolicy =
    typeof override.duplicatePolicy === "string" &&
    DUPLICATE_POLICIES.has(override.duplicatePolicy)
      ? (override.duplicatePolicy as ConferenceItemMetadataOverride["duplicatePolicy"])
      : undefined
  const tags = boundedStringArray(override.tags, MAX_KEYWORDS, MAX_KEYWORD_LENGTH)
  return {
    selected: typeof override.selected === "boolean" ? override.selected : true,
    ...(boundedString(override.title, MAX_DISPLAY_LENGTH)
      ? { title: override.title as string }
      : {}),
    ...(boundedString(override.speaker, MAX_DISPLAY_LENGTH)
      ? { speaker: override.speaker as string }
      : {}),
    ...(boundedString(override.talkDate, 64) ? { talkDate: override.talkDate as string } : {}),
    ...(boundedString(override.track, MAX_DISPLAY_LENGTH)
      ? { track: override.track as string }
      : {}),
    ...(tags ? { tags } : {}),
    ...(duplicatePolicy ? { duplicatePolicy } : {}),
  }
}

const sanitizeValidation = (value: unknown, forceInvalid: boolean): QueueItemValidation => {
  const validation = asRecord(value)
  const errors = boundedStringArray(
    validation?.errors,
    MAX_VALIDATION_MESSAGES,
    MAX_DISPLAY_LENGTH
  ) || []
  if (forceInvalid && !errors.includes("Reattach this source before processing.")) {
    errors.push("Reattach this source before processing.")
  }
  const warnings = boundedStringArray(
    validation?.warnings,
    MAX_VALIDATION_MESSAGES,
    MAX_DISPLAY_LENGTH
  )
  return {
    valid: forceInvalid
      ? false
      : validation === null
        ? true
        : typeof validation.valid === "boolean"
          ? validation.valid
          : false,
    ...(errors.length > 0 ? { errors } : {}),
    ...(warnings ? { warnings } : {}),
  }
}

const sanitizeQueueItems = (
  queueItems?: PersistedWizardQueueItem[]
): PersistedWizardQueueItem[] => {
  if (!Array.isArray(queueItems)) return []
  const detectedTypes = new Set<DetectedMediaType>([
    "audio", "video", "document", "pdf", "ebook", "image", "web", "unknown",
  ])
  const seenIds = new Set<string>()
  const sanitized: PersistedWizardQueueItem[] = []
  for (const item of queueItems) {
    if (sanitized.length >= MAX_PERSISTED_QUEUE_SOURCE_ITEMS) break
    const record = asRecord(item)
    const id = canonicalId(record?.id)
    if (!record || !id || seenIds.has(id)) continue
    seenIds.add(id)
    const sourceRef = sanitizeSourceRef(record.sourceRef, id)
    const playlist = sanitizePlaylist(record.playlist)
    const playlistReview = sanitizePlaylistReview(record.playlistReview)
    const conferenceOverride = sanitizeConferenceOverride(record.conferenceOverride)
    const hadInvalidSourceRef = record.sourceRef !== undefined && !sourceRef
    const isDirectSource = sourceRef?.kind === "direct_url"
    const isMaterializedSource = sourceRef?.kind === "materialized_playlist_item"
    const isFileSource = sourceRef?.kind === "file_stub"
    const isLegacyFile =
      !sourceRef &&
      (record.kind === "file" ||
        (!boundedString(record.url, MAX_URL_LENGTH) &&
          Boolean(
            boundedString(record.fileName, MAX_DISPLAY_LENGTH) ||
              boundedString(record.name, MAX_DISPLAY_LENGTH)
          )))
    const isFileCompatible = isFileSource || isLegacyFile
    const displayUrl = isDirectSource
      ? sourceRef.url
      : isMaterializedSource
        ? boundedString(record.url, MAX_URL_LENGTH) || playlist?.sourceUrl || undefined
        : isFileCompatible
          ? undefined
          : boundedString(record.url, MAX_URL_LENGTH)
    const missingMaterializedAuthority =
      playlistHasMaterializationCues(playlist) && !isMaterializedSource
    const detectedType = detectedTypes.has(record.detectedType as DetectedMediaType)
      ? (record.detectedType as DetectedMediaType)
      : "unknown"
    const size =
      typeof record.size === "number" && Number.isFinite(record.size) && record.size >= 0
        ? record.size
        : undefined
    const fileSize =
      typeof record.fileSize === "number" && Number.isFinite(record.fileSize) && record.fileSize >= 0
        ? record.fileSize
        : size ?? 0
    const fileStub = asRecord(record.fileStub)
    const fileName =
      boundedString(record.fileName, MAX_DISPLAY_LENGTH) ||
      boundedString(record.name, MAX_DISPLAY_LENGTH)
    const persistedLastModified =
      typeof record.lastModified === "number" && Number.isFinite(record.lastModified)
        ? record.lastModified
        : undefined
    const kind =
      isDirectSource || isMaterializedSource
        ? "url"
        : isFileCompatible
          ? "file"
          : record.kind === "url" || record.kind === "file"
            ? record.kind
            : undefined
    const next: PersistedWizardQueueItem = {
      id,
      ...(sourceRef ? { sourceRef } : {}),
      ...(kind ? { kind } : {}),
      ...(isFileCompatible && fileName ? { fileName, name: fileName } : {}),
      ...(isFileCompatible && boundedString(record.key, MAX_DISPLAY_LENGTH)
        ? { key: record.key as string }
        : {}),
      ...(isFileCompatible && size !== undefined ? { size } : {}),
      ...(isFileCompatible && boundedString(record.type, MAX_ID_LENGTH)
        ? { type: record.type as string }
        : {}),
      ...(isFileCompatible && persistedLastModified !== undefined
        ? { lastModified: persistedLastModified }
        : {}),
      ...(displayUrl ? { url: displayUrl } : {}),
      detectedType,
      icon: boundedString(record.icon, 64) || "File",
      fileSize,
      ...(boundedString(record.mimeType, MAX_ID_LENGTH)
        ? { mimeType: record.mimeType as string }
        : isFileCompatible && boundedString(record.type, MAX_ID_LENGTH)
          ? { mimeType: record.type as string }
          : {}),
      validation: sanitizeValidation(
        record.validation,
        hadInvalidSourceRef || missingMaterializedAuthority || isFileCompatible
      ),
      ...(playlist ? { playlist } : {}),
      ...(playlistReview ? { playlistReview } : {}),
      ...(conferenceOverride ? { conferenceOverride } : {}),
    }
    if (isFileCompatible && fileStub) {
      const key = boundedString(fileStub.key, MAX_DISPLAY_LENGTH)
      const instanceId = canonicalId(fileStub.instanceId)
      const lastModified =
        typeof fileStub.lastModified === "number" && Number.isFinite(fileStub.lastModified)
          ? fileStub.lastModified
          : undefined
      next.fileStub = {
        ...(key ? { key } : {}),
        ...(instanceId ? { instanceId } : {}),
        ...(lastModified !== undefined ? { lastModified } : {}),
      }
    }
    sanitized.push(next)
  }
  if (queueItems.length > MAX_PERSISTED_QUEUE_SOURCE_ITEMS) {
    const baseId = "quick-ingest-persistence-overflow"
    let overflowId = baseId
    let suffix = 1
    while (seenIds.has(overflowId)) {
      overflowId = `${baseId}-${suffix}`
      suffix += 1
    }
    sanitized.push({
      id: overflowId,
      kind: "file",
      fileName: "Incomplete restored draft",
      detectedType: "unknown",
      icon: "File",
      fileSize: 0,
      validation: {
        valid: false,
        errors: [PERSISTED_QUEUE_OVERFLOW_ERROR],
      },
    })
  }
  const queuedOccurrenceIds = new Set(sanitized.map((item) => item.id))
  return sanitized.map((item) => {
    const review = item.playlistReview
    if (!review) return item
    const nextReview = { ...review }
    let changed = false
    if (
      nextReview.duplicatePolicy &&
      nextReview.allowedDuplicatePolicies &&
      !nextReview.allowedDuplicatePolicies.includes(nextReview.duplicatePolicy)
    ) {
      delete nextReview.duplicatePolicy
      changed = true
    }
    if (nextReview.allowedDuplicatePolicies?.length === 0) {
      delete nextReview.allowedDuplicatePolicies
      changed = true
    }
    const duplicateTarget = nextReview.duplicateEvidence?.duplicateOfOccurrenceId
    if (
      nextReview.duplicateEvidence?.kind === "in_run" &&
      (!duplicateTarget || duplicateTarget === item.id || !queuedOccurrenceIds.has(duplicateTarget))
    ) {
      delete nextReview.duplicateEvidence
      delete nextReview.duplicatePolicy
      changed = true
    }
    return changed ? { ...item, playlistReview: nextReview } : item
  })
}

const ITEM_PROGRESS_STATUSES = new Set([
  "queued",
  "uploading",
  "processing",
  "analyzing",
  "storing",
  "complete",
  "failed",
  "cancelled",
])
const RUN_ITEM_STATES = new Set([
  "staged",
  "preparing",
  "awaiting_upload",
  "submit_pending",
  "queued",
  "running",
  "processing",
  "cancellation_requested",
  "status_unavailable",
  "terminal",
])
const RUN_ITEM_OUTCOMES = new Set([
  "completed",
  "included_existing",
  "metadata_updated",
  "skipped_existing",
  "submit_failed",
  "processing_failed",
  "metadata_update_failed",
  "cancelled",
])
const RESULT_OUTCOMES = new Set([
  "ingested",
  "processed",
  "skipped",
  "submit_failed",
  "failed",
  "cancelled",
])
const ERROR_CLASSIFICATIONS = new Set([
  "network",
  "auth",
  "validation",
  "server",
  "timeout",
  "unknown",
])
const PROCESSING_STATUSES = new Set([
  "idle",
  "running",
  "complete",
  "cancelled",
  "error",
])

const finiteNumber = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback

const sanitizeProcessingState = (value: unknown): WizardProcessingState => {
  const state = asRecord(value)
  const entries = Array.isArray(state?.perItemProgress)
    ? state.perItemProgress
    : []
  const perItemProgress = entries
    .slice(0, MAX_PERSISTED_SUBMISSION_OCCURRENCES)
    .flatMap((value) => {
      const progress = asRecord(value)
      const id = canonicalId(progress?.id)
      if (!progress || !id) return []
      const status = ITEM_PROGRESS_STATUSES.has(String(progress.status))
        ? (progress.status as WizardProcessingState["perItemProgress"][number]["status"])
        : "queued"
      const lifecycleState = RUN_ITEM_STATES.has(String(progress.lifecycleState))
        ? (progress.lifecycleState as NonNullable<
            WizardProcessingState["perItemProgress"][number]["lifecycleState"]
          >)
        : undefined
      const terminalOutcome = RUN_ITEM_OUTCOMES.has(String(progress.terminalOutcome))
        ? (progress.terminalOutcome as NonNullable<
            WizardProcessingState["perItemProgress"][number]["terminalOutcome"]
          >)
        : progress.terminalOutcome === null
          ? null
          : undefined
      return [{
        id,
        status,
        progressPercent: Math.min(100, Math.max(0, finiteNumber(progress.progressPercent))),
        currentStage: boundedString(progress.currentStage, MAX_DISPLAY_LENGTH) || "",
        estimatedRemaining: Math.max(0, finiteNumber(progress.estimatedRemaining)),
        ...(boundedString(progress.error, MAX_DISPLAY_LENGTH)
          ? { error: progress.error as string }
          : {}),
        ...(lifecycleState ? { lifecycleState } : {}),
        ...(terminalOutcome !== undefined ? { terminalOutcome } : {}),
        ...(typeof progress.retryable === "boolean"
          ? { retryable: progress.retryable }
          : {}),
        ...(typeof progress.attempt === "number" &&
        Number.isSafeInteger(progress.attempt) &&
        progress.attempt >= 0
          ? { attempt: Number(progress.attempt) }
          : {}),
      }]
    })
  return {
    status: PROCESSING_STATUSES.has(String(state?.status))
      ? (state?.status as WizardProcessingState["status"])
      : "idle",
    perItemProgress,
    elapsed: Math.max(0, finiteNumber(state?.elapsed)),
    estimatedRemaining: Math.max(0, finiteNumber(state?.estimatedRemaining)),
  }
}

const sanitizeResults = (value: unknown): WizardResultItem[] =>
  (Array.isArray(value) ? value : [])
    .slice(0, MAX_PERSISTED_SUBMISSION_OCCURRENCES)
    .flatMap((value) => {
      const result = asRecord(value)
      const id = canonicalId(result?.id)
      if (!result || !id) return []
      const outcome = RESULT_OUTCOMES.has(String(result.outcome))
        ? (result.outcome as WizardResultItem["outcome"])
        : undefined
      const terminalOutcome = RUN_ITEM_OUTCOMES.has(String(result.terminalOutcome))
        ? (result.terminalOutcome as NonNullable<WizardResultItem["terminalOutcome"]>)
        : result.terminalOutcome === null
          ? null
          : undefined
      const errorClassification = ERROR_CLASSIFICATIONS.has(
        String(result.errorClassification)
      )
        ? (result.errorClassification as WizardResultItem["errorClassification"])
        : undefined
      const mediaId =
        typeof result.mediaId === "number" && Number.isSafeInteger(result.mediaId)
          ? result.mediaId
          : boundedString(result.mediaId, MAX_ID_LENGTH)
      const collectionItemId =
        typeof result.collectionItemId === "number" &&
        Number.isSafeInteger(result.collectionItemId)
          ? result.collectionItemId
          : boundedString(result.collectionItemId, MAX_ID_LENGTH)
      return [{
        id,
        status: result.status === "error" ? "error" : "ok",
        ...(outcome ? { outcome } : {}),
        ...(boundedString(result.url, MAX_URL_LENGTH)
          ? { url: result.url as string }
          : {}),
        ...(boundedString(result.fileName, MAX_DISPLAY_LENGTH)
          ? { fileName: result.fileName as string }
          : {}),
        type: boundedString(result.type, MAX_ID_LENGTH) || "unknown",
        ...(boundedString(result.error, MAX_DISPLAY_LENGTH)
          ? { error: result.error as string }
          : {}),
        ...(typeof result.persisted === "boolean"
          ? { persisted: result.persisted }
          : {}),
        ...(terminalOutcome !== undefined ? { terminalOutcome } : {}),
        ...(typeof result.retryable === "boolean"
          ? { retryable: result.retryable }
          : {}),
        ...(errorClassification ? { errorClassification } : {}),
        ...(finiteNumber(result.durationMs, -1) >= 0
          ? { durationMs: finiteNumber(result.durationMs) }
          : {}),
        ...(mediaId !== undefined ? { mediaId } : {}),
        ...(collectionItemId !== undefined ? { collectionItemId } : {}),
        ...(typeof result.retryAttempt === "number" &&
        Number.isSafeInteger(result.retryAttempt) &&
        result.retryAttempt >= 0
          ? { retryAttempt: Number(result.retryAttempt) }
          : result.retryAttempt === null
            ? { retryAttempt: null }
            : {}),
        ...(boundedString(result.idempotencyKey, MAX_ID_LENGTH)
          ? { idempotencyKey: result.idempotencyKey as string }
          : result.idempotencyKey === null
            ? { idempotencyKey: null }
            : {}),
        ...(boundedString(result.title, MAX_DISPLAY_LENGTH)
          ? { title: result.title as string }
          : result.title === null
            ? { title: null }
            : {}),
        ...(boundedString(result.message, MAX_DISPLAY_LENGTH)
          ? { message: result.message as string }
          : {}),
      }]
    })

const sanitizeOpenDetail = (value: unknown): QuickIngestOpenDetail | null => {
  const detail = asRecord(value)
  if (!detail) return null
  const source = boundedString(detail.source, 64)
  const action = boundedString(detail.action, 64)
  const url = boundedString(detail.url, MAX_URL_LENGTH)
  const preferredPreset = isCustomBasePreset(detail.preferredPreset)
    ? detail.preferredPreset
    : undefined
  const sourceKind =
    detail.sourceKind === "youtube_playlist" ||
    detail.sourceKind === "youtube_watch_playlist" ||
    detail.sourceKind === "unknown"
      ? detail.sourceKind
      : undefined
  const firstSourceKind = isFirstSourceQuickIngestKind(detail.firstSourceKind)
    ? detail.firstSourceKind
    : undefined
  return {
    ...(source ? { source } : {}),
    ...(action ? { action } : {}),
    ...(url ? { url } : {}),
    ...(preferredPreset ? { preferredPreset } : {}),
    ...(sourceKind ? { sourceKind } : {}),
    ...(typeof detail.firstSource === "boolean"
      ? { firstSource: detail.firstSource }
      : {}),
    ...(firstSourceKind ? { firstSourceKind } : {}),
  }
}

const sanitizeConferenceBatchMetadata = (
  value: unknown
): ConferenceBatchMetadata | null => {
  const metadata = asRecord(value)
  const collectionName = boundedString(metadata?.collectionName, MAX_DISPLAY_LENGTH)
  if (!metadata || !collectionName) return null
  return {
    collectionName,
    ...(boundedString(metadata.conferenceName, MAX_DISPLAY_LENGTH)
      ? { conferenceName: metadata.conferenceName as string }
      : {}),
    ...(boundedString(metadata.eventDate, 64)
      ? { eventDate: metadata.eventDate as string }
      : {}),
    ...(boundedString(metadata.eventYear, 16)
      ? { eventYear: metadata.eventYear as string }
      : {}),
    sharedTags:
      boundedStringArray(metadata.sharedTags, MAX_KEYWORDS, MAX_KEYWORD_LENGTH) || [],
    ...(boundedString(metadata.sourcePlaylistUrl, MAX_URL_LENGTH)
      ? { sourcePlaylistUrl: metadata.sourcePlaylistUrl as string }
      : {}),
  }
}

const sanitizeAdvancedValues = (value: unknown): Record<string, unknown> => {
  const advanced = asRecord(value)
  if (!advanced) return {}
  const next: Record<string, unknown> = {}
  for (const key of [
    "chunk_method",
    "chunk_size",
    "chunk_overlap",
    "transcription_model",
  ]) {
    const entry = advanced[key]
    if (typeof entry === "boolean" || (typeof entry === "number" && Number.isFinite(entry))) {
      next[key] = entry
    } else if (
      typeof entry === "string" &&
      entry.length <= MAX_DISPLAY_LENGTH &&
      !entry.startsWith("data:")
    ) {
      next[key] = entry
    }
  }
  return next
}

const sanitizePresetConfig = (
  value: unknown,
  fallback: PresetConfig
): PresetConfig => {
  const config = asRecord(value)
  const common = asRecord(config?.common)
  const typeDefaults = asRecord(config?.typeDefaults)
  const audio = asRecord(typeDefaults?.audio)
  const document = asRecord(typeDefaults?.document)
  const video = asRecord(typeDefaults?.video)
  return {
    common: {
      perform_analysis:
        typeof common?.perform_analysis === "boolean"
          ? common.perform_analysis
          : fallback.common.perform_analysis,
      perform_chunking:
        typeof common?.perform_chunking === "boolean"
          ? common.perform_chunking
          : fallback.common.perform_chunking,
      overwrite_existing:
        typeof common?.overwrite_existing === "boolean"
          ? common.overwrite_existing
          : fallback.common.overwrite_existing,
      chunking_mode:
        common?.chunking_mode === "manual" || common?.chunking_mode === "auto"
          ? common.chunking_mode
          : fallback.common.chunking_mode,
      auto_chunking_goal:
        common?.auto_chunking_goal === "balanced" ||
        common?.auto_chunking_goal === "qa_search" ||
        common?.auto_chunking_goal === "navigation_summary"
          ? common.auto_chunking_goal
          : fallback.common.auto_chunking_goal,
      auto_chunking_use_llm:
        typeof common?.auto_chunking_use_llm === "boolean"
          ? common.auto_chunking_use_llm
          : fallback.common.auto_chunking_use_llm,
    },
    storeRemote:
      typeof config?.storeRemote === "boolean"
        ? config.storeRemote
        : fallback.storeRemote,
    reviewBeforeStorage:
      typeof config?.reviewBeforeStorage === "boolean"
        ? config.reviewBeforeStorage
        : fallback.reviewBeforeStorage,
    typeDefaults: {
      audio: {
        ...(boundedString(audio?.language, 64)
          ? { language: audio?.language as string }
          : {}),
        ...(typeof audio?.diarize === "boolean" ? { diarize: audio.diarize } : {}),
      },
      document: {
        ...(typeof document?.ocr === "boolean" ? { ocr: document.ocr } : {}),
      },
      video: {
        ...(typeof video?.captions === "boolean"
          ? { captions: video.captions }
          : {}),
      },
    },
    advancedValues: sanitizeAdvancedValues(config?.advancedValues),
  }
}

const sanitizeCustomOptions = (value: unknown): Partial<PresetConfig> => {
  const options = asRecord(value)
  if (!options) return {}
  const sanitized = sanitizePresetConfig(options, DEFAULT_PRESETS[DEFAULT_PRESET])
  return {
    ...(options.common ? { common: sanitized.common } : {}),
    ...(typeof options.storeRemote === "boolean"
      ? { storeRemote: options.storeRemote }
      : {}),
    ...(typeof options.reviewBeforeStorage === "boolean"
      ? { reviewBeforeStorage: options.reviewBeforeStorage }
      : {}),
    ...(options.typeDefaults ? { typeDefaults: sanitized.typeDefaults } : {}),
    ...(options.advancedValues
      ? { advancedValues: sanitized.advancedValues }
      : {}),
  }
}

const countTerminalFailures = (session: QuickIngestSessionRecord): number => {
  if (session.lifecycle === "partial_failure" || session.lifecycle === "interrupted") {
    return Math.max(
      1,
      session.resultSummary.failedCount ||
        session.results.filter((item) => item.status === "error").length
    )
  }
  return session.resultSummary.failedCount ||
    session.results.filter((item) => item.status === "error").length
}

const buildTriggerSummary = (
  session: QuickIngestSessionRecord | null
): QuickIngestTriggerSummary => {
  if (!session) {
    return {
      count: 0,
      label: null,
      hadFailure: false,
    }
  }

  const queueCount = session.queueItems.length
  const resultCount = session.results.length
  const progressCount =
    session.processingState.perItemProgress.length || queueCount || resultCount
  const failureCount = countTerminalFailures(session)
  const badgeCount = session.badge.queueCount

  switch (session.lifecycle) {
    case "draft":
      return {
        count: badgeCount || queueCount,
        label: badgeCount || queueCount ? `${badgeCount || queueCount} queued` : null,
        hadFailure: session.badge.hasRecentFailure,
      }
    case "processing":
      return {
        count: progressCount,
        label: progressCount > 0 ? `${progressCount} processing` : "Processing",
        hadFailure: false,
      }
    case "completed":
      return {
        count: resultCount || queueCount,
        label: `${resultCount || queueCount} completed`,
        hadFailure: false,
      }
    case "partial_failure":
      return {
        count: failureCount || resultCount || queueCount,
        label: `${failureCount || resultCount || queueCount} failed`,
        hadFailure: true,
      }
    case "cancelled":
      return {
        count: resultCount || queueCount,
        label: `${resultCount || queueCount} cancelled`,
        hadFailure: false,
      }
    case "interrupted":
      return {
        count: failureCount || progressCount,
        label: "Ingest interrupted",
        hadFailure: true,
      }
    default:
      return {
        count: 0,
        label: null,
        hadFailure: false,
      }
  }
}

const sanitizeSession = (
  session: QuickIngestSessionRecord | null
): QuickIngestSessionRecord | null => {
  if (!session) return null

  const createdAt =
    typeof session.createdAt === "number" && Number.isFinite(session.createdAt)
      ? session.createdAt
      : Date.now()
  const updatedAt =
    typeof session.updatedAt === "number" && Number.isFinite(session.updatedAt)
      ? session.updatedAt
      : createdAt

  const customBasePreset = isCustomBasePreset(session.customBasePreset)
    ? session.customBasePreset
    : DEFAULT_PRESET
  const selectedPreset =
    session.selectedPreset === "quick" ||
    session.selectedPreset === "standard" ||
    session.selectedPreset === "deep" ||
    session.selectedPreset === "custom"
      ? session.selectedPreset
      : DEFAULT_PRESET
  const queueItems = sanitizeQueueItems(session.queueItems)
  const results = sanitizeResults(session.results)
  const processingState = sanitizeProcessingState(session.processingState)
  const lifecycle =
    session.lifecycle === "draft" ||
    session.lifecycle === "processing" ||
    session.lifecycle === "completed" ||
    session.lifecycle === "partial_failure" ||
    session.lifecycle === "cancelled" ||
    session.lifecycle === "interrupted"
      ? session.lifecycle
      : "draft"
  const summary = asRecord(session.resultSummary)
  const summaryStatus =
    summary?.status === "idle" ||
    summary?.status === "success" ||
    summary?.status === "error" ||
    summary?.status === "cancelled"
      ? summary.status
      : "idle"
  const firstMediaId =
    typeof summary?.firstMediaId === "number" &&
    Number.isSafeInteger(summary.firstMediaId)
      ? String(summary.firstMediaId)
      : boundedString(summary?.firstMediaId, MAX_ID_LENGTH) || null

  return {
    id: session.id || generateSessionId(),
    visibility: session.visibility === "hidden" ? "hidden" : "visible",
    lifecycle,
    currentStep:
      session.currentStep === 1 ||
      session.currentStep === 2 ||
      session.currentStep === 3 ||
      session.currentStep === 4 ||
      session.currentStep === 5
        ? session.currentStep
        : 1,
    queueItems,
    selectedPreset,
    customBasePreset,
    presetConfig: sanitizePresetConfig(
      session.presetConfig,
      DEFAULT_PRESETS[customBasePreset]
    ),
    customOptions: sanitizeCustomOptions(session.customOptions),
    processingState,
    results,
    openDetail: sanitizeOpenDetail(session.openDetail),
    firstSourceAddMode: isFirstSourceQuickIngestKind(
      session.firstSourceAddMode
    )
      ? session.firstSourceAddMode
      : null,
    conferenceBatchMetadata: sanitizeConferenceBatchMetadata(
      session.conferenceBatchMetadata
    ),
    badge: {
      queueCount: Math.max(
        0,
        normalizeCountLike(session.badge?.queueCount) ?? queueItems.length
      ),
      hasRecentFailure: Boolean(session.badge?.hasRecentFailure),
    },
    resultSummary: {
      status: summaryStatus,
      attemptedAt:
        typeof summary?.attemptedAt === "number" && Number.isFinite(summary.attemptedAt)
          ? summary.attemptedAt
          : null,
      completedAt:
        typeof summary?.completedAt === "number" && Number.isFinite(summary.completedAt)
          ? summary.completedAt
          : null,
      totalCount: normalizeCountLike(summary?.totalCount) ?? 0,
      successCount: normalizeCountLike(summary?.successCount) ?? 0,
      failedCount: normalizeCountLike(summary?.failedCount) ?? 0,
      cancelledCount: normalizeCountLike(summary?.cancelledCount) ?? 0,
      firstMediaId,
      primarySourceLabel:
        boundedString(summary?.primarySourceLabel, MAX_DISPLAY_LENGTH) || null,
      errorMessage: boundedString(summary?.errorMessage, MAX_DISPLAY_LENGTH) || null,
    },
    tracking: sanitizeTracking(session.tracking),
    errorMessage: boundedString(session.errorMessage, MAX_DISPLAY_LENGTH) || null,
    createdAt,
    updatedAt,
    completedAt:
      typeof session.completedAt === "number" && Number.isFinite(session.completedAt)
        ? session.completedAt
        : null,
  }
}

const normalizeCountLike = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null
  }
  return Math.floor(value)
}

export const normalizeQuickIngestPersistenceEnvelope = (
  value: string
): string | null => {
  try {
    const envelope = asRecord(JSON.parse(value))
    const state = asRecord(envelope?.state)
    if (!state || !Object.prototype.hasOwnProperty.call(state, "session")) {
      return null
    }
    if (state.session === null) {
      return JSON.stringify({ state: { session: null }, version: 0 })
    }
    const rawSession = asRecord(state.session)
    const id = canonicalId(rawSession?.id)
    const lifecycle = rawSession?.lifecycle
    if (
      !rawSession ||
      !id ||
      (lifecycle !== "draft" &&
        lifecycle !== "processing" &&
        lifecycle !== "completed" &&
        lifecycle !== "partial_failure" &&
        lifecycle !== "cancelled" &&
        lifecycle !== "interrupted") ||
      typeof rawSession.updatedAt !== "number" ||
      !Number.isFinite(rawSession.updatedAt)
    ) {
      return null
    }
    const session = sanitizeSession(
      rawSession as unknown as QuickIngestSessionRecord
    )
    return session
      ? JSON.stringify({ state: { session }, version: 0 })
      : null
  } catch {
    return null
  }
}

const buildPersistedState = (
  session: QuickIngestSessionRecord | null
): QuickIngestSessionPersistedState => ({
  session: sanitizeSession(session),
})

const serializePersistedSession = (
  session: QuickIngestSessionRecord
): string => {
  const serialized = normalizeQuickIngestPersistenceEnvelope(
    JSON.stringify({
      state: buildPersistedState(session),
      version: 0,
    })
  )
  if (!serialized) {
    throw new TypeError("Invalid quick ingest persistence envelope")
  }
  return serialized
}

export const createEmptyQuickIngestSession = (): QuickIngestSessionRecord => {
  const now = Date.now()
  return {
    id: generateSessionId(),
    visibility: "visible",
    lifecycle: "draft",
    currentStep: 1,
    queueItems: [],
    selectedPreset: DEFAULT_PRESET,
    customBasePreset: DEFAULT_PRESET,
    presetConfig: DEFAULT_PRESETS[DEFAULT_PRESET],
    customOptions: {},
    processingState: { ...INITIAL_PROCESSING_STATE },
    results: [],
    openDetail: null,
    firstSourceAddMode: null,
    conferenceBatchMetadata: null,
    badge: {
      queueCount: 0,
      hasRecentFailure: false,
    },
    resultSummary: { ...INITIAL_RESULT_SUMMARY },
    tracking: undefined,
    errorMessage: null,
    createdAt: now,
    updatedAt: now,
    completedAt: null,
  }
}

const createInitialState = (
  persistenceStatus: QuickIngestPersistenceStatus
): QuickIngestSessionPersistedState & {
  triggerSummary: QuickIngestTriggerSummary
  persistenceStatus: QuickIngestPersistenceStatus
  isSubmissionOwner: boolean
  externalAuthorityRevision: number
} => ({
  session: null,
  triggerSummary: buildTriggerSummary(null),
  persistenceStatus,
  isSubmissionOwner: false,
  externalAuthorityRevision: 0,
})

const withSessionUpdate = (
  set: (
    partial:
      | QuickIngestSessionState
      | Partial<QuickIngestSessionState>
      | ((state: QuickIngestSessionState) => QuickIngestSessionState | Partial<QuickIngestSessionState>)
  ) => void,
  resolver: (current: QuickIngestSessionRecord | null) => QuickIngestSessionRecord | null
) => {
  set((state) => {
    const session = sanitizeSession(resolver(state.session))
    return {
      session,
      triggerSummary: buildTriggerSummary(session),
    }
  })
}

export const createQuickIngestSessionStore = (options: {
  persistence?: QuickIngestIndexedDbStorage
} = {}) => {
  const persistence =
    options.persistence ||
    createQuickIngestIndexedDbStorage({
      normalizeValue: normalizeQuickIngestPersistenceEnvelope,
    })
  let lastBackgroundValue: string | null = null
  const backgroundStorage: StateStorage = {
    getItem: async (name) => {
      try {
        return await persistence.storage.getItem(name)
      } catch {
        return null
      }
    },
    setItem: async (name, value) => {
      try {
        const envelope = JSON.parse(value) as {
          state?: QuickIngestSessionPersistedState
        }
        if (!envelope.state?.session) return
      } catch {
        return
      }
      if (value === lastBackgroundValue) return
      lastBackgroundValue = value
      try {
        await persistence.storage.setItem(name, value)
      } catch {
        if (lastBackgroundValue === value) lastBackgroundValue = null
        // The adapter publishes the visible failure status.
      }
    },
    removeItem: async (name) => {
      lastBackgroundValue = null
      try {
        await persistence.storage.removeItem(name)
      } catch {
        // The adapter publishes the visible failure status.
      }
    },
  }
  const store = createWithEqualityFn<QuickIngestSessionState>()(
    persist(
      (set, get) => ({
        ...createInitialState(persistence.getStatus()),
        createDraftSession: (seed) => {
          const next = sanitizeSession({
            ...createEmptyQuickIngestSession(),
            ...(seed || {}),
            updatedAt: Date.now(),
          })
          set({
            session: next,
            triggerSummary: buildTriggerSummary(next),
          })
          return next as QuickIngestSessionRecord
        },
        upsertSession: (next) =>
          withSessionUpdate(set, (current) => {
            const base = current || createEmptyQuickIngestSession()
            return {
              ...base,
              ...next,
              badge: {
                ...base.badge,
                ...(next.badge || {}),
              },
              resultSummary: {
                ...base.resultSummary,
                ...(next.resultSummary || {}),
              },
              tracking:
                next.lifecycle === "draft"
                  ? undefined
                  : next.tracking === undefined
                  ? base.tracking
                  : mergeTracking(base.tracking, next.tracking),
              updatedAt: Date.now(),
            }
          }),
        showSession: () =>
          withSessionUpdate(set, (current) => {
            if (!current) {
              return createEmptyQuickIngestSession()
            }
            return {
              ...current,
              visibility: "visible",
              updatedAt: Date.now(),
            }
          }),
        hideSession: () =>
          withSessionUpdate(set, (current) => {
            if (!current) return current
            return {
              ...current,
              visibility: "hidden",
              updatedAt: Date.now(),
            }
          }),
        markProcessingTracking: (tracking) =>
          withSessionUpdate(set, (current) => {
            const base = current || createEmptyQuickIngestSession()
            return {
              ...base,
              lifecycle: "processing",
              tracking: mergeTracking(base.tracking, {
                ...tracking,
                startedAt:
                  tracking.startedAt || base.tracking?.startedAt || Date.now(),
              }),
              updatedAt: Date.now(),
            }
          }),
        clearProcessingTracking: () =>
          withSessionUpdate(set, (current) => {
            if (!current) return current
            return {
              ...current,
              tracking: undefined,
              updatedAt: Date.now(),
            }
          }),
        commitReviewHandoff: async (next) => {
          const state = get()
          const current = state.session
          if (!current) return false
          const authorityRevision = state.externalAuthorityRevision
          const expectedValue = serializePersistedSession(current)
          const session = sanitizeSession({
            ...current,
            ...next,
            badge: {
              ...current.badge,
              ...(next.badge || {}),
            },
            resultSummary: {
              ...current.resultSummary,
              ...(next.resultSummary || {}),
            },
            tracking: undefined,
            updatedAt: Date.now(),
          })
          if (!session) return false
          try {
            await persistence.initialize()
            const committed = await persistence.commitReviewHandoff(
              expectedValue,
              serializePersistedSession(session)
            )
            if (!committed) return false
          } catch {
            return false
          }
          const live = get()
          if (
            !live.session ||
            live.session.id !== current.id ||
            live.externalAuthorityRevision !== authorityRevision ||
            serializePersistedSession(live.session) !== expectedValue
          ) {
            return false
          }
          set({
            session,
            triggerSummary: buildTriggerSummary(session),
          })
          return true
        },
        commitProcessingHandoff: async (next, tracking) => {
          const state = get()
          const current = state.session
          if (
            !current ||
            state.persistenceStatus !== "ready" ||
            !state.isSubmissionOwner ||
            current.lifecycle !== "draft"
          ) {
            return false
          }
          const authorityRevision = state.externalAuthorityRevision
          const expectedValue = serializePersistedSession(current)
          const session = sanitizeSession({
            ...current,
            ...next,
            lifecycle: "processing",
            currentStep: 4,
            badge: {
              ...current.badge,
              ...(next.badge || {}),
            },
            resultSummary: {
              ...current.resultSummary,
              ...(next.resultSummary || {}),
            },
            tracking: mergeTracking(current.tracking, {
              ...tracking,
              submissionState: "creating_run",
              startedAt:
                tracking.startedAt || current.tracking?.startedAt || Date.now(),
            }),
            updatedAt: Date.now(),
          })
          if (!session) return false
          try {
            await persistence.initialize()
            const committed = await persistence.commitProcessingHandoff(
              expectedValue,
              serializePersistedSession(session)
            )
            if (!committed) return false
          } catch {
            return false
          }
          const live = get()
          if (
            !live.session ||
            live.session.id !== current.id ||
            live.externalAuthorityRevision !== authorityRevision ||
            live.persistenceStatus !== "ready" ||
            !live.isSubmissionOwner ||
            serializePersistedSession(live.session) !== expectedValue
          ) {
            return false
          }
          set({
            session,
            triggerSummary: buildTriggerSummary(session),
          })
          return true
        },
        acquireSubmissionLease: async () => {
          const sessionId = get().session?.id
          if (!sessionId || get().persistenceStatus !== "ready") {
            set({ isSubmissionOwner: false })
            return false
          }
          let acquired = false
          try {
            acquired = await persistence.acquireSubmissionLease(sessionId)
            const value = await persistence.storage.getItem(STORAGE_KEY)
            const normalized = value
              ? normalizeQuickIngestPersistenceEnvelope(value)
              : null
            if (normalized) {
              const envelope = JSON.parse(normalized) as {
                state?: QuickIngestSessionPersistedState
              }
              const durableSession = envelope.state?.session
              const currentSession = get().session
              if (
                durableSession?.id === sessionId &&
                currentSession?.id === sessionId &&
                (durableSession.lifecycle !== "draft" ||
                  durableSession.updatedAt > currentSession.updatedAt)
              ) {
                set((state) => ({
                  session: durableSession,
                  triggerSummary: buildTriggerSummary(durableSession),
                  isSubmissionOwner: acquired,
                  externalAuthorityRevision:
                    state.externalAuthorityRevision + 1,
                }))
                return acquired
              }
            }
            set({ isSubmissionOwner: acquired })
            return acquired
          } catch {
            if (acquired) {
              await persistence.releaseSubmissionLease(sessionId).catch(() => {})
            }
            set({ isSubmissionOwner: false })
            return false
          }
        },
        renewSubmissionLease: async () => {
          const sessionId = get().session?.id
          if (!sessionId || get().persistenceStatus !== "ready") {
            set({ isSubmissionOwner: false })
            return false
          }
          try {
            const renewed = await persistence.renewSubmissionLease(sessionId)
            set({ isSubmissionOwner: renewed })
            return renewed
          } catch {
            set({ isSubmissionOwner: false })
            return false
          }
        },
        releaseSubmissionLease: async () => {
          const sessionId = get().session?.id
          set({ isSubmissionOwner: false })
          if (!sessionId) return
          try {
            await persistence.releaseSubmissionLease(sessionId)
          } catch {
            // The persistence status subscription exposes the failure.
          }
        },
        markInterrupted: (reason) =>
          withSessionUpdate(set, (current) => {
            if (!current) return current
            return {
              ...current,
              lifecycle: "interrupted",
              badge: {
                ...current.badge,
                hasRecentFailure: true,
              },
              resultSummary: {
                ...current.resultSummary,
                status: "error",
                errorMessage: reason || "Quick ingest was interrupted.",
              },
              errorMessage: reason || "Quick ingest was interrupted.",
              updatedAt: Date.now(),
            }
          }),
        clearSession: () =>
          {
            const session = get().session
            set({
              session: null,
              triggerSummary: buildTriggerSummary(null),
              isSubmissionOwner: false,
            })
            if (!session) return
            if (session.lifecycle === "draft") {
              void backgroundStorage.removeItem(STORAGE_KEY)
              return
            }
            void persistence
              .clearAuthoritativeSession({
                id: session.id,
                lifecycle: session.lifecycle,
                updatedAt: session.updatedAt,
              })
              .catch(() => {})
          },
        replaceWithNewDraft: (seed) => {
          get().clearSession()
          return get().createDraftSession(seed)
        },
      }),
      {
        name: STORAGE_KEY,
        // Baseline version so future shape changes can migrate instead of discarding
        // persisted state (see apps/FRONTEND_AUDIT.md §6 / TASK-12102).
        version: 1,
        migrate: (persisted) => persisted as any,
        storage: createJSONStorage(() => backgroundStorage),
        partialize: (state) => buildPersistedState(state.session),
        merge: (persistedState, currentState) => {
          const nextSession = sanitizeSession(
            (persistedState as QuickIngestSessionPersistedState | undefined)?.session ||
              null
          )
          const currentSession = currentState.session
          const mergedSession =
            nextSession &&
            nextSession.lifecycle !== "draft" &&
            currentSession?.lifecycle === "draft"
              ? nextSession
              : currentSession &&
                  (!nextSession || currentSession.updatedAt > nextSession.updatedAt)
              ? currentSession
              : nextSession
          return {
            ...currentState,
            session: mergedSession,
            triggerSummary: buildTriggerSummary(mergedSession),
          }
        },
      }
    )
  )

  persistence.subscribeStatus((persistenceStatus) => {
    store.setState({
      persistenceStatus,
      ...(persistenceStatus === "ready" ? {} : { isSubmissionOwner: false }),
    })
  })
  void persistence.initialize().catch(() => {})
  return store
}

export const useQuickIngestSessionStore = createQuickIngestSessionStore()

if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  // Expose for debugging/tests only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useQuickIngestSessionStore = useQuickIngestSessionStore
}
