import type {
  ScrapedItem,
  ScrapedItemSortMode,
  WatchlistItemSavedViewCreate,
  WatchlistItemSavedViewFilters,
  WatchlistSource
} from "@/types/watchlists"

export const SOURCE_LOAD_PAGE_SIZE = 200
export const SOURCE_LOAD_MAX_ITEMS = 1000
export const SOURCE_LIST_INITIAL_RENDER_COUNT = 120
export const SOURCE_LIST_RENDER_CHUNK = 120
export const SOURCE_LIST_SCROLL_EXPAND_THRESHOLD_PX = 180
export const ITEM_PAGE_SIZE = 25
export const ITEM_PAGE_SIZE_OPTIONS = [20, 25, 50, 100] as const
export const ITEMS_PAGE_SIZE_STORAGE_KEY = "watchlists:items:page-size"
export const ITEMS_SORT_MODE_STORAGE_KEY = "watchlists:items:sort-mode"
export const ITEMS_VIEW_PRESETS_STORAGE_KEY = "watchlists:items:view-presets"
export const DEFAULT_ITEMS_SORT_MODE = "newest"

export type ItemsSortMode = "newest" | "oldest" | "unreadFirst" | "reviewedFirst"

export const READER_SORT_MODES = ["newest", "oldest", "unreadFirst"] as const
export type ReaderSortMode = (typeof READER_SORT_MODES)[number]

export const SYSTEM_ITEMS_VIEW_PRESET_IDS = {
  unreadToday: "system-unread-today",
  highPriority: "system-high-priority",
  needsReview: "system-needs-review"
} as const

export type SystemItemsViewPresetId =
  typeof SYSTEM_ITEMS_VIEW_PRESET_IDS[keyof typeof SYSTEM_ITEMS_VIEW_PRESET_IDS]

export interface PersistedItemsViewPreset {
  id: string
  name: string
  sourceId: number | null
  smartFilter: string
  statusFilter: string
  sortMode?: string
  searchQuery: string
}

export const buildDefaultItemsViewPresets = (
  labels?: Partial<Record<SystemItemsViewPresetId, string>>
): PersistedItemsViewPreset[] => [
  {
    id: SYSTEM_ITEMS_VIEW_PRESET_IDS.unreadToday,
    name: labels?.[SYSTEM_ITEMS_VIEW_PRESET_IDS.unreadToday] || "Unread today",
    sourceId: null,
    smartFilter: "todayUnread",
    statusFilter: "ingested",
    sortMode: "unreadFirst",
    searchQuery: ""
  },
  {
    id: SYSTEM_ITEMS_VIEW_PRESET_IDS.highPriority,
    name: labels?.[SYSTEM_ITEMS_VIEW_PRESET_IDS.highPriority] || "High-priority",
    sourceId: null,
    smartFilter: "todayUnread",
    statusFilter: "ingested",
    sortMode: "unreadFirst",
    searchQuery: "urgent"
  },
  {
    id: SYSTEM_ITEMS_VIEW_PRESET_IDS.needsReview,
    name: labels?.[SYSTEM_ITEMS_VIEW_PRESET_IDS.needsReview] || "Needs review",
    sourceId: null,
    smartFilter: "unread",
    statusFilter: "all",
    sortMode: "newest",
    searchQuery: ""
  }
]

export const normalizeItemsSortMode = (value: unknown): ItemsSortMode => {
  if (
    value === "newest" ||
    value === "oldest" ||
    value === "unreadFirst" ||
    value === "reviewedFirst"
  ) {
    return value
  }
  return DEFAULT_ITEMS_SORT_MODE
}

export const toServerItemSortMode = (value: unknown): ScrapedItemSortMode => {
  const normalized = normalizeItemsSortMode(value)
  if (normalized === "oldest") return "created_asc"
  if (normalized === "unreadFirst") return "unread_first"
  return "created_desc"
}

const uniquePresetsById = (
  presets: PersistedItemsViewPreset[]
): PersistedItemsViewPreset[] => {
  const seenIds = new Set<string>()
  const deduped: PersistedItemsViewPreset[] = []
  for (const preset of presets) {
    if (seenIds.has(preset.id)) continue
    seenIds.add(preset.id)
    deduped.push(preset)
  }
  return deduped
}

export const provisionItemsViewPresets = (
  persistedPresets: PersistedItemsViewPreset[],
  defaultPresets: PersistedItemsViewPreset[]
): PersistedItemsViewPreset[] => {
  const normalizedPersisted = uniquePresetsById(persistedPresets)
  const persistedById = new Map(
    normalizedPersisted.map((preset) => [preset.id, preset] as const)
  )
  const defaultIds = new Set(defaultPresets.map((preset) => preset.id))

  const mergedDefaults = defaultPresets.map((preset) => {
    const persistedMatch = persistedById.get(preset.id)
    if (!persistedMatch) return preset
    return { ...preset, ...persistedMatch, id: preset.id }
  })

  const customPresets = normalizedPersisted
    .filter((preset) => !defaultIds.has(preset.id))
    .map((preset) => ({
      ...preset,
      sortMode: normalizeItemsSortMode(preset.sortMode)
    }))
    .sort((left, right) => {
      const byName = left.name.localeCompare(right.name)
      if (byName !== 0) return byName
      return left.id.localeCompare(right.id)
    })

  return [...mergedDefaults, ...customPresets]
}

export const isSystemItemsViewPresetId = (presetId: string | null | undefined): boolean => {
  if (!presetId) return false
  return Object.values(SYSTEM_ITEMS_VIEW_PRESET_IDS).includes(
    presetId as SystemItemsViewPresetId
  )
}

export const normalizeItemPageSize = (value: unknown): number => {
  const parsed = Number(value)
  if (ITEM_PAGE_SIZE_OPTIONS.includes(parsed as typeof ITEM_PAGE_SIZE_OPTIONS[number])) {
    return parsed
  }
  return ITEM_PAGE_SIZE
}

export const normalizeReaderSortMode = (value: unknown): ReaderSortMode => {
  const normalized = String(value || "")
  if (
    normalized === "newest" ||
    normalized === "oldest" ||
    normalized === "unreadFirst"
  ) {
    return normalized
  }
  return "newest"
}

export const loadPersistedItemPageSize = (
  storage: Pick<Storage, "getItem"> | null | undefined
): number => {
  try {
    const raw = storage?.getItem(ITEMS_PAGE_SIZE_STORAGE_KEY)
    if (raw == null) return ITEM_PAGE_SIZE
    return normalizeItemPageSize(raw)
  } catch {
    return ITEM_PAGE_SIZE
  }
}

export const persistItemPageSize = (
  storage: Pick<Storage, "setItem"> | null | undefined,
  pageSize: number
): void => {
  try {
    storage?.setItem(ITEMS_PAGE_SIZE_STORAGE_KEY, String(normalizeItemPageSize(pageSize)))
  } catch {
    // Ignore storage write errors (private browsing, quota, etc.)
  }
}

export const loadPersistedItemsSortMode = (
  storage: Pick<Storage, "getItem"> | null | undefined
): ReaderSortMode => {
  try {
    const raw = storage?.getItem(ITEMS_SORT_MODE_STORAGE_KEY)
    if (raw == null) return "newest"
    return normalizeReaderSortMode(raw)
  } catch {
    return "newest"
  }
}

export const persistItemsSortMode = (
  storage: Pick<Storage, "setItem"> | null | undefined,
  sortMode: ReaderSortMode
): void => {
  try {
    storage?.setItem(ITEMS_SORT_MODE_STORAGE_KEY, normalizeReaderSortMode(sortMode))
  } catch {
    // Ignore storage write errors (private browsing, quota, etc.)
  }
}

const isPersistedItemsViewPreset = (value: unknown): value is PersistedItemsViewPreset => {
  if (!value || typeof value !== "object") return false
  const candidate = value as Record<string, unknown>
  if (typeof candidate.id !== "string" || candidate.id.trim().length === 0) return false
  if (typeof candidate.name !== "string" || candidate.name.trim().length === 0) return false
  const sourceId = candidate.sourceId
  if (sourceId !== null && typeof sourceId !== "number") return false
  if (typeof candidate.smartFilter !== "string") return false
  if (typeof candidate.statusFilter !== "string") return false
  if (candidate.sortMode != null && typeof candidate.sortMode !== "string") return false
  if (typeof candidate.searchQuery !== "string") return false
  return true
}

export const loadPersistedItemsViewPresets = (
  storage: Pick<Storage, "getItem"> | null | undefined
): PersistedItemsViewPreset[] => {
  try {
    const raw = storage?.getItem(ITEMS_VIEW_PRESETS_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isPersistedItemsViewPreset)
  } catch {
    return []
  }
}

export const persistItemsViewPresets = (
  storage: Pick<Storage, "setItem"> | null | undefined,
  presets: PersistedItemsViewPreset[]
): void => {
  try {
    storage?.setItem(
      ITEMS_VIEW_PRESETS_STORAGE_KEY,
      JSON.stringify(presets.filter(isPersistedItemsViewPreset))
    )
  } catch {
    // Ignore storage write errors (private browsing, quota, etc.)
  }
}

export const buildServerItemViewCreatePayload = (
  preset: PersistedItemsViewPreset
): WatchlistItemSavedViewCreate => {
  const filters: WatchlistItemSavedViewFilters = {}
  if (preset.sourceId != null) {
    filters.source_id = preset.sourceId
  }
  if (preset.smartFilter && preset.smartFilter !== "all") {
    filters.smart_filter = preset.smartFilter as WatchlistItemSavedViewFilters["smart_filter"]
  }
  if (preset.statusFilter && preset.statusFilter !== "all") {
    filters.status = preset.statusFilter
  }
  const query = preset.searchQuery.trim()
  if (query) {
    filters.q = query
  }
  return {
    name: preset.name.trim(),
    filters,
    sort: toServerItemSortMode(preset.sortMode),
    is_default: false
  }
}

export const getMigratableItemsViewPresets = (
  storage: Pick<Storage, "getItem"> | null | undefined
): PersistedItemsViewPreset[] =>
  loadPersistedItemsViewPresets(storage).filter((preset) => !isSystemItemsViewPresetId(preset.id))

export const filterSourcesForReader = (
  sources: WatchlistSource[],
  query: string
): WatchlistSource[] => {
  const trimmed = query.trim().toLowerCase()
  if (!trimmed) return sources

  return sources.filter((source) => {
    if (source.name.toLowerCase().includes(trimmed)) return true
    if (source.url.toLowerCase().includes(trimmed)) return true
    return source.tags.some((tag) => tag.toLowerCase().includes(trimmed))
  })
}

export const getInitialSourceRenderCount = (
  totalSources: number,
  searchQuery: string
): number => {
  const normalizedTotal = Number.isFinite(totalSources)
    ? Math.max(0, Math.floor(totalSources))
    : 0
  if (normalizedTotal === 0) return 0
  if (searchQuery.trim().length > 0) return normalizedTotal
  return Math.min(normalizedTotal, SOURCE_LIST_INITIAL_RENDER_COUNT)
}

export const getNextSourceRenderCount = (
  currentCount: number,
  totalSources: number
): number => {
  const normalizedTotal = Number.isFinite(totalSources)
    ? Math.max(0, Math.floor(totalSources))
    : 0
  const normalizedCurrent = Number.isFinite(currentCount)
    ? Math.max(0, Math.floor(currentCount))
    : 0

  if (normalizedCurrent >= normalizedTotal) return normalizedTotal
  return Math.min(normalizedTotal, normalizedCurrent + SOURCE_LIST_RENDER_CHUNK)
}

export const shouldExpandSourceRenderWindow = (
  scrollTop: number,
  scrollHeight: number,
  clientHeight: number
): boolean => {
  if (
    !Number.isFinite(scrollTop) ||
    !Number.isFinite(scrollHeight) ||
    !Number.isFinite(clientHeight)
  ) {
    return false
  }
  if (scrollHeight <= 0 || clientHeight <= 0) return false

  const remaining = scrollHeight - (scrollTop + clientHeight)
  return remaining <= SOURCE_LIST_SCROLL_EXPAND_THRESHOLD_PX
}

const sourcePriorityTimestamp = (source: WatchlistSource): number => {
  if (!source.last_scraped_at) return Number.NEGATIVE_INFINITY
  const parsed = new Date(source.last_scraped_at).getTime()
  if (Number.isNaN(parsed)) return Number.NEGATIVE_INFINITY
  return parsed
}

const sourceHealthBucket = (source: WatchlistSource): number => {
  const status = source.status?.trim().toLowerCase() || ""
  if (!status) return 2
  if (status === "healthy" || status === "ok") return 2
  return 1
}

export const orderSourcesForReader = (
  sources: WatchlistSource[],
  selectedSourceId: number | null
): WatchlistSource[] => {
  return [...sources].sort((left, right) => {
    const leftSelected = selectedSourceId != null && left.id === selectedSourceId
    const rightSelected = selectedSourceId != null && right.id === selectedSourceId
    if (leftSelected !== rightSelected) return leftSelected ? -1 : 1

    const leftActiveBucket = left.active ? 0 : 1
    const rightActiveBucket = right.active ? 0 : 1
    if (leftActiveBucket !== rightActiveBucket) {
      return leftActiveBucket - rightActiveBucket
    }

    const leftHealth = sourceHealthBucket(left)
    const rightHealth = sourceHealthBucket(right)
    if (leftHealth !== rightHealth) return leftHealth - rightHealth

    const leftTimestamp = sourcePriorityTimestamp(left)
    const rightTimestamp = sourcePriorityTimestamp(right)
    if (leftTimestamp !== rightTimestamp) return rightTimestamp - leftTimestamp

    return left.name.localeCompare(right.name)
  })
}

export const resolveSelectedItemId = (
  currentId: number | null,
  items: ScrapedItem[]
): number | null => {
  if (items.length === 0) return null
  if (currentId && items.some((item) => item.id === currentId)) return currentId
  return items[0].id
}

const toItemTimestamp = (item: ScrapedItem): number => {
  const candidate = item.published_at || item.created_at || ""
  const parsed = new Date(candidate).getTime()
  if (Number.isNaN(parsed)) return Number.NEGATIVE_INFINITY
  return parsed
}

export const sortItemsForReader = (
  items: ScrapedItem[],
  sortMode: ReaderSortMode
): ScrapedItem[] => {
  const normalizedMode = normalizeReaderSortMode(sortMode)
  return [...items].sort((left, right) => {
    const leftTs = toItemTimestamp(left)
    const rightTs = toItemTimestamp(right)

    if (normalizedMode === "unreadFirst" && left.reviewed !== right.reviewed) {
      return left.reviewed ? 1 : -1
    }

    if (normalizedMode === "oldest") {
      if (leftTs !== rightTs) return leftTs - rightTs
    } else {
      if (leftTs !== rightTs) return rightTs - leftTs
    }

    return left.id - right.id
  })
}

const decodeCommonEntities = (value: string): string => {
  return value
    .split("&nbsp;").join(" ")
    .split("&NBSP;").join(" ")
    .split("&lt;").join("<")
    .split("&LT;").join("<")
    .split("&gt;").join(">")
    .split("&GT;").join(">")
    .split("&amp;").join("&")
    .split("&AMP;").join("&")
}

const stripHtmlTagsWithoutRegex = (value: string): string => {
  let output = ""
  let insideTag = false
  for (let idx = 0; idx < value.length; idx += 1) {
    const ch = value[idx]
    if (ch === "<") {
      insideTag = true
      continue
    }
    if (ch === ">") {
      if (insideTag) output += " "
      insideTag = false
      continue
    }
    if (!insideTag) output += ch
  }
  return output
}

export const stripHtmlToText = (value: string): string => {
  if (!value) return ""
  if (typeof DOMParser !== "undefined") {
    try {
      const doc = new DOMParser().parseFromString(value, "text/html")
      doc.querySelectorAll("script, style").forEach((node) => node.remove())
      return (doc.body?.textContent || "").replace(/\s+/g, " ").trim()
    } catch {
      // Fall through to non-DOM fallback.
    }
  }
  return decodeCommonEntities(stripHtmlTagsWithoutRegex(value))
    .replace(/\s+/g, " ")
    .trim()
}

export const shouldReloadItemsAfterReviewMutation = (
  smartFilter: string
): boolean => {
  return (
    smartFilter === "unread" ||
    smartFilter === "reviewed" ||
    smartFilter === "todayUnread"
  )
}

export const extractImageUrl = (value: string | null | undefined): string | null => {
  if (!value) return null

  const htmlMatch = value.match(/<img[^>]+src=["']([^"']+)["']/i)
  if (htmlMatch?.[1]) return htmlMatch[1]

  const markdownMatch = value.match(/!\[[^\]]*]\((https?:\/\/[^)\s]+)\)/i)
  if (markdownMatch?.[1]) return markdownMatch[1]

  return null
}
