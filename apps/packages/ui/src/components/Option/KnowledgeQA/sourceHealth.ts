import {
  getRagSourceLabel,
  isRagSource,
} from "@/services/rag/sourceMetadata"
import { READY_STATE_LABEL, UNAVAILABLE_STATE_LABEL } from "@/design-system"
import type {
  KnowledgeSourceEmbeddingStatus,
  KnowledgeSourceHealth,
  KnowledgeSourceHealthState,
  KnowledgeSourceIndexStatus,
} from "./types"

const INDEX_STATUSES = new Set<KnowledgeSourceIndexStatus>([
  "ready",
  "indexing",
  "stale",
  "empty",
  "unavailable",
  "error",
  "unknown",
])

const EMBEDDING_STATUSES = new Set<KnowledgeSourceEmbeddingStatus>([
  "ready",
  "indexing",
  "missing",
  "unavailable",
  "not_applicable",
  "error",
  "unknown",
])

export const EMPTY_SOURCE_HEALTH_STATE: KnowledgeSourceHealthState = {
  bySource: {},
  sources: [],
  loading: false,
  error: null,
  loadedAt: null,
}

const toRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const toOptionalNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null

const toOptionalString = (value: unknown): string | null =>
  typeof value === "string" && value.trim().length > 0 ? value : null

export function normalizeIndexStatus(value: unknown): KnowledgeSourceIndexStatus {
  return typeof value === "string" && INDEX_STATUSES.has(value as KnowledgeSourceIndexStatus)
    ? (value as KnowledgeSourceIndexStatus)
    : "unknown"
}

export function normalizeEmbeddingStatus(
  value: unknown
): KnowledgeSourceEmbeddingStatus {
  return typeof value === "string" &&
    EMBEDDING_STATUSES.has(value as KnowledgeSourceEmbeddingStatus)
    ? (value as KnowledgeSourceEmbeddingStatus)
    : "unknown"
}

export function normalizeKnowledgeSourceHealth(
  payload: unknown,
  personalLists?: Partial<Record<"media_db" | "notes", unknown>>
): KnowledgeSourceHealthState {
  const record = toRecord(payload) ?? {}
  const rawSources = Array.isArray(record.sources) ? record.sources : []
  const sources: KnowledgeSourceHealth[] = []

  for (const raw of rawSources) {
    const entry = toRecord(raw)
    const sourceId = isRagSource(entry?.source_id) ? entry.source_id : null
    if (!sourceId) {
      continue
    }

    const list = toRecord(personalLists?.[sourceId as "media_db" | "notes"])
    const pagination = toRecord(list?.pagination)
    const total = toOptionalNumber(
      pagination?.total ?? pagination?.total_items ?? list?.total
    )
    sources.push({
      sourceId,
      label: toOptionalString(entry?.label) ?? getRagSourceLabel(sourceId),
      available: entry?.available === true,
      searchable: entry?.searchable === true,
      itemCount:
        total != null && Number.isSafeInteger(total) && total >= 0
          ? total
          : toOptionalNumber(entry?.item_count),
      indexedCount: toOptionalNumber(entry?.indexed_count),
      lastUpdated: toOptionalString(entry?.last_updated),
      lastIndexed: toOptionalString(entry?.last_indexed),
      indexStatus: normalizeIndexStatus(entry?.index_status),
      embeddingStatus: normalizeEmbeddingStatus(entry?.embedding_status),
      disabledReason: toOptionalString(entry?.disabled_reason),
      workspaceScoped: entry?.workspace_scoped === true,
      hiddenByDefault: entry?.hidden_by_default === true,
      privacyNote: toOptionalString(entry?.privacy_note),
    })
  }

  return {
    bySource: Object.fromEntries(
      sources.map((source) => [source.sourceId, source])
    ),
    sources,
    loading: false,
    error: null,
    loadedAt: new Date().toISOString(),
  }
}

export function getSourceHealthStatusLabel(
  health: KnowledgeSourceHealth | null | undefined
): string {
  if (!health) {
    return "Unknown"
  }
  if (health.workspaceScoped) {
    return "Workspace only"
  }
  switch (health.indexStatus) {
    case "ready":
      return health.searchable ? READY_STATE_LABEL : UNAVAILABLE_STATE_LABEL
    case "indexing":
      return "Indexing"
    case "stale":
      return "Stale"
    case "empty":
      return "Empty"
    case "unavailable":
      return UNAVAILABLE_STATE_LABEL
    case "error":
      return "Error"
    case "unknown":
    default:
      return "Unknown"
  }
}

export function getSourceHealthChipClass(
  health: KnowledgeSourceHealth | null | undefined
): string {
  const base =
    "inline-flex shrink-0 items-center rounded-full border px-1.5 py-0.5 text-[10px] font-medium"
  if (!health) {
    return `${base} border-border bg-surface2 text-text-muted`
  }
  if (health.workspaceScoped) {
    return `${base} border-info/30 bg-info/10 text-info`
  }
  switch (health.indexStatus) {
    case "ready":
      return health.searchable
        ? `${base} border-success/30 bg-success/10 text-success`
        : `${base} border-warn/30 bg-warn/10 text-warn`
    case "indexing":
      return `${base} border-info/30 bg-info/10 text-info`
    case "stale":
    case "empty":
      return `${base} border-warn/30 bg-warn/10 text-warn`
    case "unavailable":
    case "error":
      return `${base} border-danger/30 bg-danger/10 text-danger`
    case "unknown":
    default:
      return `${base} border-border bg-surface2 text-text-muted`
  }
}

export function buildSourceHealthSummary(
  state: KnowledgeSourceHealthState | null | undefined
): string {
  if (!state || state.error) {
    return "Source health unavailable"
  }
  if (state.loading) {
    return "Checking source health..."
  }
  if (state.sources.length === 0) {
    return "Source health unavailable"
  }

  const availableCount = state.sources.filter(
    (source) => source.available
  ).length
  const personalCount = getPersonalItemCount(state)
  const personalSources = [state.bySource.media_db, state.bySource.notes]
  const searchableCount =
    personalCount === 0
      ? 0
      : personalSources.every(
            (source) =>
              source &&
              source.indexedCount != null &&
              Number.isSafeInteger(source.indexedCount) &&
              source.indexedCount >= 0
          )
        ? personalSources.reduce(
            (total, source) =>
              total + (source?.searchable ? (source.indexedCount ?? 0) : 0),
            0
          )
        : "unknown"
  const content = state.personalContentError
    ? "Personal items: unavailable"
    : personalCount == null
      ? "Personal items: unknown"
      : `Stored personal items: ${personalCount} · Searchable personal items: ${searchableCount}`
  return `Available services: ${availableCount} of ${state.sources.length} · ${content}`
}

/** Storage totals never imply embeddings exist. Unknown/loading counts stay unknown. */
export function getPersonalItemCount(
  state: KnowledgeSourceHealthState | null | undefined
): number | null {
  if (!state || state.loading || state.error || state.personalContentError)
    return null
  const counts = [
    state.bySource.media_db?.itemCount,
    state.bySource.notes?.itemCount
  ]
  return counts.every(
    (count) => count != null && Number.isSafeInteger(count) && count >= 0
  )
    ? counts.reduce<number>((total, count) => total + (count ?? 0), 0)
    : null
}
