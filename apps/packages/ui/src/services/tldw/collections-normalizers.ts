import type {
  ReadingDigestSchedule,
  ReadingDigestScheduleFilters,
  ReadingDigestSuggestionStatus
} from "@/types/collections"
import type {
  IngestionSourceItem,
  IngestionSourceItemsListResponse,
  IngestionSourceListResponse,
  IngestionSourceSummary,
  IngestionSourceSyncSummary,
  IngestionSourceSyncTriggerResponse
} from "@/types/ingestion-sources"

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const READING_DIGEST_STATUSES = new Set<ReadingDigestSuggestionStatus>([
  "saved",
  "reading",
  "read",
  "archived"
])

const READING_DIGEST_SORTS = new Set<
  NonNullable<ReadingDigestScheduleFilters["sort"]>
>([
  "updated_desc",
  "updated_asc",
  "created_desc",
  "created_asc",
  "title_asc",
  "title_desc",
  "relevance"
])

const toNullableFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return null
}

const toOptionalFiniteNumber = (value: unknown): number | undefined => {
  const normalized = toNullableFiniteNumber(value)
  return normalized === null ? undefined : normalized
}

const normalizeStringArray = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined
  }
  return value
    .map((entry) => String(entry ?? "").trim())
    .filter((entry) => entry.length > 0)
}

const normalizeReadingDigestStatusArray = (
  value: unknown
): ReadingDigestSuggestionStatus[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined
  }

  return value
    .map((entry) => String(entry ?? "").trim())
    .filter(
      (entry): entry is ReadingDigestSuggestionStatus =>
        READING_DIGEST_STATUSES.has(entry as ReadingDigestSuggestionStatus)
    )
}

const normalizeReadingDigestFilters = (
  value: unknown
): ReadingDigestSchedule["filters"] => {
  if (!isObjectRecord(value)) {
    return null
  }

  const filters: ReadingDigestScheduleFilters = {}
  const status = normalizeReadingDigestStatusArray(value.status)
  const tags = normalizeStringArray(value.tags)
  const suggestions = isObjectRecord(value.suggestions)
    ? {
        enabled: Boolean(value.suggestions.enabled),
        ...(toOptionalFiniteNumber(value.suggestions.limit) !== undefined
          ? { limit: toOptionalFiniteNumber(value.suggestions.limit) }
          : {}),
        ...(normalizeReadingDigestStatusArray(value.suggestions.status) !==
        undefined
          ? { status: normalizeReadingDigestStatusArray(value.suggestions.status) }
          : {}),
        ...(normalizeStringArray(value.suggestions.exclude_tags) !== undefined
          ? { exclude_tags: normalizeStringArray(value.suggestions.exclude_tags) }
          : {}),
        ...(toOptionalFiniteNumber(value.suggestions.max_age_days) !== undefined
          ? { max_age_days: toOptionalFiniteNumber(value.suggestions.max_age_days) }
          : {}),
        ...(typeof value.suggestions.include_read === "boolean"
          ? { include_read: value.suggestions.include_read }
          : {}),
        ...(typeof value.suggestions.include_archived === "boolean"
          ? { include_archived: value.suggestions.include_archived }
          : {})
      }
    : undefined

  if (status !== undefined) {
    filters.status = status
  }
  if (tags !== undefined) {
    filters.tags = tags
  }
  if (typeof value.favorite === "boolean") {
    filters.favorite = value.favorite
  }
  if (value.domain != null) {
    filters.domain = String(value.domain)
  }
  if (value.q != null) {
    filters.q = String(value.q)
  }
  if (value.date_from != null) {
    filters.date_from = String(value.date_from)
  }
  if (value.date_to != null) {
    filters.date_to = String(value.date_to)
  }
  if (
    typeof value.sort === "string" &&
    READING_DIGEST_SORTS.has(
      value.sort as NonNullable<ReadingDigestScheduleFilters["sort"]>
    )
  ) {
    filters.sort = value.sort as ReadingDigestScheduleFilters["sort"]
  }
  if (toOptionalFiniteNumber(value.limit) !== undefined) {
    filters.limit = toOptionalFiniteNumber(value.limit)
  }
  if (suggestions !== undefined) {
    filters.suggestions = suggestions
  }

  return filters
}

export const normalizeReadingDigestSchedule = (
  schedule: unknown
): ReadingDigestSchedule => {
  const source = isObjectRecord(schedule) ? schedule : {}

  return {
    id: String(source.id ?? ""),
    name: source.name == null ? null : String(source.name),
    cron: String(source.cron ?? ""),
    timezone: source.timezone == null ? null : String(source.timezone),
    enabled: Boolean(source.enabled),
    require_online: Boolean(source.require_online),
    format: source.format === "html" ? "html" : "md",
    template_id: toNullableFiniteNumber(source.template_id),
    template_name:
      source.template_name == null ? null : String(source.template_name),
    retention_days: toNullableFiniteNumber(source.retention_days),
    filters: normalizeReadingDigestFilters(source.filters),
    last_run_at: source.last_run_at == null ? null : String(source.last_run_at),
    next_run_at: source.next_run_at == null ? null : String(source.next_run_at),
    last_status: source.last_status == null ? null : String(source.last_status),
    created_at: source.created_at == null ? null : String(source.created_at),
    updated_at: source.updated_at == null ? null : String(source.updated_at)
  }
}

export const toFiniteNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return fallback
}

export const toOptionalString = (value: unknown): string | null => {
  if (value === null || typeof value === "undefined") {
    return null
  }
  return String(value)
}

export const toRecord = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {}
  }
  return { ...(value as Record<string, unknown>) }
}

export const normalizeIngestionSourceSyncSummary = (
  summary: unknown
): IngestionSourceSyncSummary | null => {
  if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
    return null
  }
  const record = summary as Record<string, unknown>
  return {
    changed_count: toFiniteNumber(record.changed_count),
    degraded_count: toFiniteNumber(record.degraded_count),
    conflict_count: toFiniteNumber(record.conflict_count),
    sink_failure_count: toFiniteNumber(record.sink_failure_count),
    ingestion_failure_count: toFiniteNumber(record.ingestion_failure_count),
    created_count: toFiniteNumber(record.created_count),
    updated_count: toFiniteNumber(record.updated_count),
    deleted_count: toFiniteNumber(record.deleted_count),
    unchanged_count: toFiniteNumber(record.unchanged_count)
  }
}

export const normalizeIngestionSourceType = (
  value: unknown
): IngestionSourceSummary["source_type"] => {
  if (value === "archive_snapshot" || value === "git_repository") {
    return value
  }
  return "local_directory"
}

export const normalizeIngestionSource = (
  source: any
): IngestionSourceSummary => ({
  id: String(source?.id ?? ""),
  user_id: toFiniteNumber(source?.user_id),
  source_type: normalizeIngestionSourceType(source?.source_type),
  sink_type: source?.sink_type === "notes" ? "notes" : "media",
  policy: source?.policy === "import_only" ? "import_only" : "canonical",
  enabled: Boolean(source?.enabled),
  schedule_enabled: Boolean(source?.schedule_enabled),
  schedule_config: toRecord(source?.schedule_config),
  config: toRecord(source?.config),
  active_job_id: toOptionalString(source?.active_job_id),
  last_successful_snapshot_id: toOptionalString(
    source?.last_successful_snapshot_id
  ),
  last_sync_started_at: toOptionalString(source?.last_sync_started_at),
  last_sync_completed_at: toOptionalString(source?.last_sync_completed_at),
  last_sync_status: toOptionalString(source?.last_sync_status),
  last_error: toOptionalString(source?.last_error),
  last_successful_sync_summary: normalizeIngestionSourceSyncSummary(
    source?.last_successful_sync_summary
  ),
  created_at: toOptionalString(source?.created_at),
  updated_at: toOptionalString(source?.updated_at)
})

export const normalizeIngestionSourceListResponse = (
  payload: any
): IngestionSourceListResponse => {
  const rawSources = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.sources)
      ? payload.sources
      : []
  const sources = rawSources.map((source: any) =>
    normalizeIngestionSource(source)
  )
  return {
    sources,
    total: toFiniteNumber(payload?.total, sources.length)
  }
}

export const normalizeIngestionSourceItem = (
  item: any
): IngestionSourceItem => ({
  id: String(item?.id ?? ""),
  source_id: String(item?.source_id ?? ""),
  normalized_relative_path: String(item?.normalized_relative_path ?? ""),
  content_hash: item?.content_hash == null ? null : String(item.content_hash),
  sync_status: String(item?.sync_status ?? "unknown"),
  binding: toRecord(item?.binding),
  present_in_source: Boolean(item?.present_in_source),
  created_at: toOptionalString(item?.created_at),
  updated_at: toOptionalString(item?.updated_at)
})

export const normalizeIngestionSourceItemsListResponse = (
  payload: any
): IngestionSourceItemsListResponse => {
  const rawItems = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.items)
      ? payload.items
      : []
  const items = rawItems.map((item: any) => normalizeIngestionSourceItem(item))
  return {
    items,
    total: toFiniteNumber(payload?.total, items.length)
  }
}

export const normalizeIngestionSourceSyncTrigger = (
  payload: any
): IngestionSourceSyncTriggerResponse => ({
  status: String(payload?.status ?? ""),
  source_id: String(payload?.source_id ?? ""),
  job_id: toOptionalString(payload?.job_id),
  snapshot_status: toOptionalString(payload?.snapshot_status)
})
