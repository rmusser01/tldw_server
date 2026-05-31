import type {
  ConferenceBatchMetadata,
  ConferenceDuplicatePolicy,
  ConferenceItemMetadataOverride,
  PlaylistQueueMetadata,
  WizardResultItem,
} from "@/components/Common/QuickIngest/types"

export type MediaCollectionItemStatus =
  | "planned"
  | "processing"
  | "completed"
  | "skipped_existing"
  | "submit_failed"
  | "failed"
  | "cancelled"

export type ApiMediaCollectionItem = {
  id?: number
  collection_id?: number
  ordinal?: number
  source_url?: string
  normalized_source_id?: string | null
  source_kind?: string | null
  title?: string | null
  speaker?: string | null
  published_at?: string | null
  track?: string | null
  duplicate_status?: string | null
  status?: string | null
  media_id?: number | null
  content_item_id?: number | null
  latest_job_id?: string | null
  latest_run_id?: number | null
  idempotency_key?: string | null
  retry_count?: number
  error_summary?: string | null
  warnings?: string[]
  metadata?: Record<string, unknown>
  tags?: string[]
  created_at?: string
  updated_at?: string
}

export type ApiMediaCollection = {
  id?: number
  name?: string
  kind?: string
  description?: string | null
  source_url?: string | null
  metadata?: Record<string, unknown>
  default_tags?: string[]
  created_at?: string
  updated_at?: string
  items?: ApiMediaCollectionItem[]
}

export type ApiMediaCollectionListResponse = {
  items?: ApiMediaCollection[]
  total?: number
  page?: number
  size?: number
}

export type MediaCollectionItem = {
  id: number
  collectionId: number
  ordinal: number
  sourceUrl: string
  normalizedSourceId: string | null
  sourceKind: string | null
  title: string | null
  speaker: string | null
  publishedAt: string | null
  track: string | null
  duplicateStatus: string
  status: MediaCollectionItemStatus
  mediaId: number | null
  contentItemId: number | null
  latestJobId: string | null
  latestRunId: number | null
  idempotencyKey: string | null
  retryCount: number
  errorSummary: string | null
  warnings: string[]
  metadata: Record<string, unknown>
  tags: string[]
  createdAt: string
  updatedAt: string
}

export type MediaCollection = {
  id: number
  name: string
  kind: string
  description: string | null
  sourceUrl: string | null
  metadata: Record<string, unknown>
  defaultTags: string[]
  createdAt: string
  updatedAt: string
  items: MediaCollectionItem[]
}

export type MediaCollectionList = {
  items: MediaCollection[]
  total: number
  page: number
  size: number
}

export type MediaCollectionStatusCounts = {
  total: number
  planned: number
  processing: number
  completed: number
  skippedExisting: number
  submitFailed: number
  failed: number
  cancelled: number
}

export type ConferenceDuplicatePolicyResolution = {
  policy: ConferenceDuplicatePolicy
  plannedStatus: MediaCollectionItemStatus
  shouldSubmitJob: boolean
  forceOverwrite: boolean
}

export type ConferenceIngestFailureKind =
  | "auth_required"
  | "unavailable"
  | "timeout"
  | "cancelled"
  | "validation"
  | "server"
  | "unknown"

export type ConferenceRetryRequestItem = {
  resultId: string
  collectionItemId: string
  retryAttempt: number
  idempotencyKey: string
}

export type ConferenceCollectionCreatePayload = {
  name: string
  kind: "conference"
  description?: string | null
  source_url?: string | null
  metadata: Record<string, unknown>
  default_tags: string[]
}

export type ConferenceCollectionItemPayload = {
  ordinal?: number
  source_url: string
  normalized_source_id?: string | null
  source_kind?: string | null
  title?: string | null
  speaker?: string | null
  published_at?: string | null
  track?: string | null
  duplicate_status?: string | null
  status?: MediaCollectionItemStatus
  idempotency_key?: string | null
  metadata: Record<string, unknown>
  tags: string[]
}

export type ConferenceCollectionItemMergeInput = {
  id: string
  url: string
  playlist?: PlaylistQueueMetadata
  conferenceOverride?: ConferenceItemMetadataOverride
}

const compactString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed || undefined
}

export const CONFERENCE_DUPLICATE_POLICIES: ConferenceDuplicatePolicy[] = [
  "skip",
  "overwrite",
  "update_metadata_only",
  "include_existing",
]

export const normalizeConferenceDuplicatePolicy = (
  value: unknown
): ConferenceDuplicatePolicy =>
  CONFERENCE_DUPLICATE_POLICIES.includes(value as ConferenceDuplicatePolicy)
    ? (value as ConferenceDuplicatePolicy)
    : "skip"

const isDuplicateStatus = (value: unknown): boolean => {
  const normalized = compactString(value)?.toLowerCase()
  return normalized === "duplicate_existing" || normalized === "duplicate_in_batch"
}

export const resolveConferenceDuplicatePolicy = (
  duplicateStatus: unknown,
  policy: unknown
): ConferenceDuplicatePolicyResolution => {
  const resolvedPolicy = normalizeConferenceDuplicatePolicy(policy)
  if (!isDuplicateStatus(duplicateStatus)) {
    return {
      policy: resolvedPolicy,
      plannedStatus: "planned",
      shouldSubmitJob: true,
      forceOverwrite: false,
    }
  }

  if (resolvedPolicy === "overwrite") {
    return {
      policy: resolvedPolicy,
      plannedStatus: "planned",
      shouldSubmitJob: true,
      forceOverwrite: true,
    }
  }

  return {
    policy: resolvedPolicy,
    plannedStatus: "skipped_existing",
    shouldSubmitJob: false,
    forceOverwrite: false,
  }
}

export const classifyConferenceIngestFailure = (
  value: unknown
): ConferenceIngestFailureKind => {
  const message = String(value || "").trim().toLowerCase()
  if (!message) return "unknown"
  if (
    message.includes("private") ||
    message.includes("sign in") ||
    message.includes("login") ||
    message.includes("unauthorized") ||
    message.includes("forbidden") ||
    message.includes("403")
  ) {
    return "auth_required"
  }
  if (
    message.includes("404") ||
    message.includes("not found") ||
    message.includes("unavailable") ||
    message.includes("removed")
  ) {
    return "unavailable"
  }
  if (
    message.includes("timeout") ||
    message.includes("timed out") ||
    message.includes("deadline")
  ) {
    return "timeout"
  }
  if (message.includes("cancelled") || message.includes("canceled")) {
    return "cancelled"
  }
  if (message.includes("invalid") || message.includes("unsupported")) {
    return "validation"
  }
  if (message.includes("500") || message.includes("server")) {
    return "server"
  }
  return "unknown"
}

const isRetryableConferenceOutcome = (
  item: Pick<WizardResultItem, "status" | "outcome">
): boolean =>
  item.outcome === "submit_failed" ||
  item.outcome === "failed" ||
  item.outcome === "cancelled" ||
  item.status === "error"

export const buildConferenceRetryRequestItems = (
  items: WizardResultItem[]
): ConferenceRetryRequestItem[] =>
  items
    .filter(isRetryableConferenceOutcome)
    .map((item) => {
      const collectionItemId =
        item.collectionItemId == null
          ? undefined
          : compactString(String(item.collectionItemId))
      if (!collectionItemId) return null
      const retryAttempt =
        typeof item.retryAttempt === "number" && Number.isFinite(item.retryAttempt)
          ? Math.trunc(item.retryAttempt) + 1
          : 1
      return {
        resultId: item.id,
        collectionItemId,
        retryAttempt,
        idempotencyKey: `conference-retry-${collectionItemId}-${retryAttempt}`,
      }
    })
    .filter((item): item is ConferenceRetryRequestItem => Boolean(item))

export const normalizeConferenceTagList = (value: unknown): string[] => {
  const raw = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(",")
      : []
  return Array.from(
    new Set(
      raw
        .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
        .filter(Boolean)
    )
  )
}

export const mergeConferenceTags = (
  sharedTags: unknown,
  itemTags: unknown
): string[] =>
  normalizeConferenceTagList([
    ...normalizeConferenceTagList(sharedTags),
    ...normalizeConferenceTagList(itemTags),
  ])

export const buildConferenceCollectionCreatePayload = (
  metadata: ConferenceBatchMetadata,
  fallbackName?: string | null
): ConferenceCollectionCreatePayload => {
  const sourcePlaylistUrl = compactString(metadata.sourcePlaylistUrl)
  const collectionMetadata: Record<string, unknown> = {}
  const conferenceName = compactString(metadata.conferenceName)
  const eventDate = compactString(metadata.eventDate)
  const eventYear = compactString(metadata.eventYear)

  if (conferenceName) collectionMetadata.conference_name = conferenceName
  if (eventDate) collectionMetadata.event_date = eventDate
  if (eventYear) collectionMetadata.event_year = eventYear
  if (sourcePlaylistUrl) {
    collectionMetadata.source_playlist_url = sourcePlaylistUrl
  }

  return {
    name:
      compactString(metadata.collectionName) ||
      conferenceName ||
      compactString(fallbackName) ||
      "Conference batch",
    kind: "conference",
    source_url: sourcePlaylistUrl ?? null,
    metadata: collectionMetadata,
    default_tags: normalizeConferenceTagList(metadata.sharedTags),
  }
}

export const buildConferenceCollectionItemPayload = (
  batchMetadata: ConferenceBatchMetadata,
  item: ConferenceCollectionItemMergeInput
): ConferenceCollectionItemPayload => {
  const override = item.conferenceOverride
  const playlist = item.playlist
  const duplicatePolicy = normalizeConferenceDuplicatePolicy(
    override?.duplicatePolicy
  )
  const duplicateResolution = resolveConferenceDuplicatePolicy(
    playlist?.duplicateStatus,
    duplicatePolicy
  )
  const ordinal = playlist?.ordinal
  const metadata: Record<string, unknown> = {
    quick_ingest_item_id: item.id,
  }
  const playlistId = compactString(playlist?.playlistId ?? undefined)
  const playlistTitle = compactString(playlist?.playlistTitle ?? undefined)
  if (playlistId) metadata.playlist_id = playlistId
  if (playlistTitle) metadata.playlist_title = playlistTitle
  if (ordinal != null) metadata.playlist_ordinal = ordinal
  if (compactString(batchMetadata.eventYear)) {
    metadata.event_year = compactString(batchMetadata.eventYear)
  }
  if (isDuplicateStatus(playlist?.duplicateStatus)) {
    metadata.duplicate_policy = duplicatePolicy
  }

  return {
    ordinal,
    source_url: item.url,
    normalized_source_id: compactString(playlist?.normalizedSourceId ?? undefined) ?? null,
    source_kind: playlistId ? "youtube_video" : null,
    title: compactString(override?.title) ?? null,
    speaker: compactString(override?.speaker) ?? null,
    published_at:
      compactString(override?.talkDate) ??
      compactString(batchMetadata.eventDate) ??
      null,
    track: compactString(override?.track) ?? null,
    duplicate_status: compactString(playlist?.duplicateStatus ?? undefined) ?? "unknown",
    status: duplicateResolution.plannedStatus,
    metadata,
    tags: mergeConferenceTags(batchMetadata.sharedTags, override?.tags),
  }
}

const nullableString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const finiteNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value)
  }
  return fallback
}

const finiteNumberOrNull = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value)
  }
  return null
}

const normalizeRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? { ...(value as Record<string, unknown>) }
    : {}

const normalizeStringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0)
    : []

export const normalizeMediaCollectionItemStatus = (
  value: unknown
): MediaCollectionItemStatus => {
  if (
    value === "planned" ||
    value === "processing" ||
    value === "completed" ||
    value === "skipped_existing" ||
    value === "submit_failed" ||
    value === "failed" ||
    value === "cancelled"
  ) {
    return value
  }
  return "planned"
}

export const normalizeMediaCollectionItem = (
  payload: ApiMediaCollectionItem
): MediaCollectionItem => ({
  id: finiteNumber(payload?.id),
  collectionId: finiteNumber(payload?.collection_id),
  ordinal: finiteNumber(payload?.ordinal, 1),
  sourceUrl: nullableString(payload?.source_url) || "",
  normalizedSourceId: nullableString(payload?.normalized_source_id),
  sourceKind: nullableString(payload?.source_kind),
  title: nullableString(payload?.title),
  speaker: nullableString(payload?.speaker),
  publishedAt: nullableString(payload?.published_at),
  track: nullableString(payload?.track),
  duplicateStatus: nullableString(payload?.duplicate_status) || "unknown",
  status: normalizeMediaCollectionItemStatus(payload?.status),
  mediaId: finiteNumberOrNull(payload?.media_id),
  contentItemId: finiteNumberOrNull(payload?.content_item_id),
  latestJobId: nullableString(payload?.latest_job_id),
  latestRunId: finiteNumberOrNull(payload?.latest_run_id),
  idempotencyKey: nullableString(payload?.idempotency_key),
  retryCount: finiteNumber(payload?.retry_count),
  errorSummary: nullableString(payload?.error_summary),
  warnings: normalizeStringList(payload?.warnings),
  metadata: normalizeRecord(payload?.metadata),
  tags: normalizeStringList(payload?.tags),
  createdAt: nullableString(payload?.created_at) || "",
  updatedAt: nullableString(payload?.updated_at) || ""
})

export const normalizeMediaCollectionResponse = (
  payload: ApiMediaCollection
): MediaCollection => ({
  id: finiteNumber(payload?.id),
  name: nullableString(payload?.name) || "Untitled collection",
  kind: nullableString(payload?.kind) || "conference",
  description: nullableString(payload?.description),
  sourceUrl: nullableString(payload?.source_url),
  metadata: normalizeRecord(payload?.metadata),
  defaultTags: normalizeStringList(payload?.default_tags),
  createdAt: nullableString(payload?.created_at) || "",
  updatedAt: nullableString(payload?.updated_at) || "",
  items: Array.isArray(payload?.items)
    ? payload.items.map((item) => normalizeMediaCollectionItem(item))
    : []
})

export const normalizeMediaCollectionListResponse = (
  payload: ApiMediaCollectionListResponse
): MediaCollectionList => {
  const items = Array.isArray(payload?.items)
    ? payload.items.map((item) => normalizeMediaCollectionResponse(item))
    : []
  return {
    items,
    total: finiteNumber(payload?.total, items.length),
    page: finiteNumber(payload?.page, 1),
    size: finiteNumber(payload?.size, items.length || 20)
  }
}

export const getMediaCollectionStatusCounts = (
  collection: Pick<MediaCollection, "items">
): MediaCollectionStatusCounts => {
  const counts: MediaCollectionStatusCounts = {
    total: collection.items.length,
    planned: 0,
    processing: 0,
    completed: 0,
    skippedExisting: 0,
    submitFailed: 0,
    failed: 0,
    cancelled: 0
  }
  for (const item of collection.items) {
    if (item.status === "skipped_existing") {
      counts.skippedExisting += 1
    } else if (item.status === "submit_failed") {
      counts.submitFailed += 1
    } else {
      counts[item.status] += 1
    }
  }
  return counts
}

export const getConferenceResultExportStatus = (
  item: Pick<WizardResultItem, "outcome" | "status">
): string => {
  if (item.outcome === "submit_failed") return "submit_failed"
  if (item.outcome === "cancelled") return "cancelled"
  if (item.outcome === "failed" || item.status === "error") return "failed"
  if (item.outcome === "skipped") return "skipped_existing"
  return item.outcome || "completed"
}

export const buildConferenceFailedResultExportText = (
  items: WizardResultItem[]
): string =>
  items
    .map((item, index) => {
      const title =
        compactString(item.title) ||
        compactString(item.fileName) ||
        compactString(item.url) ||
        item.id
      const lines = [`#${index + 1}`, `Title: ${title}`]
      const url = compactString(item.url)
      const collectionItemId =
        item.collectionItemId == null
          ? undefined
          : compactString(String(item.collectionItemId))
      const retryAttempt =
        typeof item.retryAttempt === "number" && Number.isFinite(item.retryAttempt)
          ? Math.trunc(item.retryAttempt)
          : null
      const errorSummary = compactString(item.error) || compactString(item.message)

      if (url) lines.push(`URL: ${url}`)
      if (collectionItemId) lines.push(`Collection item: ${collectionItemId}`)
      lines.push(`Status: ${getConferenceResultExportStatus(item)}`)
      if (retryAttempt != null) lines.push(`Retry attempt: ${retryAttempt}`)
      if (errorSummary) lines.push(`Error: ${errorSummary}`)

      return lines.join("\n")
    })
    .join("\n\n")
