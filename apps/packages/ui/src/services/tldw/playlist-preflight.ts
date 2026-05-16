export type PlaylistDuplicateStatus =
  | "new"
  | "duplicate_in_batch"
  | "duplicate_existing"
  | "unknown"

export type ApiPlaylistPreflightItem = {
  ordinal?: number
  source_url?: string
  normalized_source_id?: string | null
  source_kind?: string
  title?: string | null
  speaker?: string | null
  duration_seconds?: number | null
  published_at?: string | null
  thumbnail_url?: string | null
  duplicate_status?: PlaylistDuplicateStatus | string | null
  duplicate_of_ordinal?: number | null
  selected?: boolean | null
}

export type ApiPlaylistPreflightResponse = {
  source_url?: string
  source_kind?: string
  playlist_id?: string | null
  playlist_title?: string | null
  video_id?: string | null
  item_count?: number
  selected_count?: number
  duplicate_count?: number
  warnings?: string[]
  items?: ApiPlaylistPreflightItem[]
}

export type PlaylistPreflightItem = {
  id: string
  ordinal: number
  sourceUrl: string
  normalizedSourceId: string | null
  sourceKind: string
  title: string
  speaker: string | null
  durationSeconds: number | null
  publishedAt: string | null
  thumbnailUrl: string | null
  duplicateStatus: PlaylistDuplicateStatus
  duplicateOfOrdinal: number | null
  selected: boolean
}

export type PlaylistPreflightResult = {
  sourceUrl: string
  sourceKind: string
  playlistId: string | null
  playlistTitle: string | null
  videoId: string | null
  itemCount: number
  selectedCount: number
  duplicateCount: number
  warnings: string[]
  items: PlaylistPreflightItem[]
}

const normalizeDuplicateStatus = (value: unknown): PlaylistDuplicateStatus => {
  if (
    value === "new" ||
    value === "duplicate_in_batch" ||
    value === "duplicate_existing" ||
    value === "unknown"
  ) {
    return value
  }
  return "unknown"
}

const nullableString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const finiteNumberOrNull = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  return value
}

export const isPlaylistPreflightDuplicate = (
  item: Pick<PlaylistPreflightItem, "duplicateStatus">
): boolean => item.duplicateStatus !== "new"

export const normalizePlaylistPreflightResponse = (
  payload: ApiPlaylistPreflightResponse
): PlaylistPreflightResult => {
  const items = Array.isArray(payload?.items) ? payload.items : []
  const normalizedItems = items.map((item, index) => {
    const ordinal =
      typeof item?.ordinal === "number" && Number.isFinite(item.ordinal) && item.ordinal > 0
        ? Math.floor(item.ordinal)
        : index + 1
    const sourceUrl = nullableString(item?.source_url) || ""
    const normalizedSourceId = nullableString(item?.normalized_source_id)
    const duplicateStatus = normalizeDuplicateStatus(item?.duplicate_status)
    const id = normalizedSourceId || sourceUrl || `playlist-item-${ordinal}`
    return {
      id,
      ordinal,
      sourceUrl,
      normalizedSourceId,
      sourceKind: nullableString(item?.source_kind) || "generic_url",
      title: nullableString(item?.title) || sourceUrl || `Item ${ordinal}`,
      speaker: nullableString(item?.speaker),
      durationSeconds: finiteNumberOrNull(item?.duration_seconds),
      publishedAt: nullableString(item?.published_at),
      thumbnailUrl: nullableString(item?.thumbnail_url),
      duplicateStatus,
      duplicateOfOrdinal: finiteNumberOrNull(item?.duplicate_of_ordinal),
      selected:
        typeof item?.selected === "boolean"
          ? item.selected
          : duplicateStatus === "new"
    }
  })

  return {
    sourceUrl: nullableString(payload?.source_url) || "",
    sourceKind: nullableString(payload?.source_kind) || "generic_url",
    playlistId: nullableString(payload?.playlist_id),
    playlistTitle: nullableString(payload?.playlist_title),
    videoId: nullableString(payload?.video_id),
    itemCount:
      typeof payload?.item_count === "number" && Number.isFinite(payload.item_count)
        ? payload.item_count
        : normalizedItems.length,
    selectedCount:
      typeof payload?.selected_count === "number" && Number.isFinite(payload.selected_count)
        ? payload.selected_count
        : normalizedItems.filter((item) => item.selected).length,
    duplicateCount:
      typeof payload?.duplicate_count === "number" && Number.isFinite(payload.duplicate_count)
        ? payload.duplicate_count
        : normalizedItems.filter(isPlaylistPreflightDuplicate).length,
    warnings: Array.isArray(payload?.warnings)
      ? payload.warnings.filter((item): item is string => typeof item === "string")
      : [],
    items: normalizedItems
  }
}
