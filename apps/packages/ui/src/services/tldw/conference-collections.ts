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
