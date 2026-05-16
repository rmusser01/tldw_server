import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Checkbox,
  Button,
  Empty,
  Input,
  Modal,
  Pagination,
  Progress,
  Select,
  Segmented,
  Space,
  Spin,
  Tag,
  Tooltip,
  message
} from "antd"
import DOMPurify from "dompurify"
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  ExternalLink,
  HelpCircle,
  MessageSquare,
  RefreshCw,
  Rss,
  Sun
} from "lucide-react"
import { useTranslation } from "react-i18next"
import type { TFunction } from "i18next"
import {
  batchUpdateScrapedItems,
  createWatchlistItemView,
  createWatchlistOutput,
  deleteWatchlistItemView,
  fetchScrapedItemSmartCounts,
  fetchScrapedItems,
  fetchWatchlistItemViews,
  fetchWatchlistRuns,
  fetchWatchlistSources,
  updateWatchlistItemView,
  updateScrapedItem
} from "@/services/watchlists"
import type { FetchItemsParams } from "@/services/watchlists"
import { useWatchlistsStore } from "@/store/watchlists"
import type {
  ScrapedItem,
  ScrapedItemAlertSummary,
  ScrapedItemBatchScope,
  ScrapedItemBatchUpdateResponse,
  WatchlistContentAlertSeverity,
  WatchlistItemSavedView,
  WatchlistItemSavedViewFilters,
  WatchlistRun,
  WatchlistSource
} from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import {
  buildServerItemViewCreatePayload,
  buildDefaultItemsViewPresets,
  DEFAULT_ITEMS_SORT_MODE,
  extractImageUrl,
  filterSourcesForReader,
  getMigratableItemsViewPresets,
  getInitialSourceRenderCount,
  getNextSourceRenderCount,
  ITEM_PAGE_SIZE_OPTIONS,
  isSystemItemsViewPresetId,
  loadPersistedItemsSortMode,
  loadPersistedItemPageSize,
  loadPersistedItemsViewPresets,
  normalizeReaderSortMode,
  orderSourcesForReader,
  persistItemsSortMode,
  persistItemPageSize,
  persistItemsViewPresets,
  provisionItemsViewPresets,
  type PersistedItemsViewPreset,
  type ReaderSortMode,
  resolveSelectedItemId,
  shouldReloadItemsAfterReviewMutation,
  shouldExpandSourceRenderWindow,
  SOURCE_LIST_INITIAL_RENDER_COUNT,
  SOURCE_LOAD_MAX_ITEMS,
  SOURCE_LOAD_PAGE_SIZE,
  SYSTEM_ITEMS_VIEW_PRESET_IDS,
  stripHtmlToText,
  toServerItemSortMode
} from "./items-utils"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"
import { useNavigate } from "react-router-dom"
import { setSetting } from "@/services/settings"
import { DISCUSS_WATCHLIST_PROMPT_SETTING } from "@/services/settings/ui-settings"
import {
  type WatchlistChatHandoffPayload,
  type WatchlistChatArticle,
  getWatchlistChatTotalChars,
  WATCHLIST_CHAT_CONTENT_WARN_THRESHOLD
} from "@/services/tldw/watchlist-chat-handoff"

const { Search } = Input

type ReaderStatusFilter = "all" | "ingested" | "filtered"
type SmartFeedFilter = "all" | "today" | "todayUnread" | "unread" | "reviewed" | "queued" | "alertMatches"
type BatchReviewScope = "selected" | "page" | "allFiltered"
type BatchReviewPhase = "running" | "complete" | "partial" | "failed"
type ItemsViewPreset = Omit<PersistedItemsViewPreset, "smartFilter" | "statusFilter" | "sortMode"> & {
  smartFilter: SmartFeedFilter
  statusFilter: ReaderStatusFilter
  sortMode: ReaderSortMode
  serverId?: number
}
const SHORTCUTS_HINT_DISMISSED_STORAGE_KEY = "watchlists:items:shortcuts-hint-dismissed"
const SMART_COUNTS_CACHE_TTL_MS = 15_000

interface SmartCountsCacheEntry {
  counts: Record<SmartFeedFilter, number>
  cachedAt: number
}

interface BatchReviewProgress {
  scope: BatchReviewScope
  phase: BatchReviewPhase
  isRunning: boolean
  total: number
  processed: number
  succeeded: number
  failed: number
  incomplete: boolean
  failedIds: number[]
  failedItemIds: number[]
}

const normalizeSmartFilter = (value: string): SmartFeedFilter => {
  if (value === "today_unread") return "todayUnread"
  if (
    value === "today" ||
    value === "todayUnread" ||
    value === "unread" ||
    value === "reviewed" ||
    value === "queued" ||
    value === "alertMatches"
  ) {
    return value
  }
  return "all"
}

const normalizeStatusFilter = (value: string): ReaderStatusFilter => {
  if (value === "ingested" || value === "filtered") return value
  return "all"
}

const toReaderSortModeFromServer = (value: string | null | undefined): ReaderSortMode => {
  if (value === "created_asc" || value === "published_asc") return "oldest"
  if (value === "unread_first") return "unreadFirst"
  return DEFAULT_ITEMS_SORT_MODE
}

const getSmartFilterFromServerFilters = (
  filters: WatchlistItemSavedViewFilters
): SmartFeedFilter => {
  if (filters.smart_filter) return normalizeSmartFilter(filters.smart_filter)
  if (filters.has_alert) return "alertMatches"
  if (filters.queued_for_briefing) return "queued"
  if (filters.reviewed === true) return "reviewed"
  if (filters.reviewed === false) return "unread"
  return "all"
}

const toItemsViewPresetFromServer = (view: WatchlistItemSavedView): ItemsViewPreset => ({
  id: `server-${view.id}`,
  serverId: view.id,
  name: view.name,
  sourceId: typeof view.filters.source_id === "number" ? view.filters.source_id : null,
  smartFilter: getSmartFilterFromServerFilters(view.filters),
  statusFilter: normalizeStatusFilter(String(view.filters.status || "all")),
  sortMode: toReaderSortModeFromServer(view.sort),
  searchQuery: String(view.filters.q || view.filters.search || "")
})

const makeCurrentViewPreset = (
  name: string,
  sourceId: number | null,
  smartFilter: SmartFeedFilter,
  statusFilter: ReaderStatusFilter,
  sortMode: ReaderSortMode,
  searchQuery: string
): PersistedItemsViewPreset => ({
  id: "current",
  name,
  sourceId,
  smartFilter,
  statusFilter,
  sortMode,
  searchQuery
})

const severityRank: Record<WatchlistContentAlertSeverity, number> = {
  info: 0,
  low: 1,
  medium: 2,
  high: 3,
  critical: 4
}

const severityTagColor: Record<WatchlistContentAlertSeverity, string> = {
  info: "blue",
  low: "default",
  medium: "gold",
  high: "orange",
  critical: "red"
}

const formatSeverityLabel = (
  severity: WatchlistContentAlertSeverity | null | undefined
): string => {
  if (!severity) return "Alert"
  return severity.charAt(0).toUpperCase() + severity.slice(1)
}

const getHighestSeverity = (
  summary: ScrapedItemAlertSummary | null | undefined
): WatchlistContentAlertSeverity | null => {
  if (!summary) return null
  if (summary.highest_severity) return summary.highest_severity
  return (summary.severities || []).reduce<WatchlistContentAlertSeverity | null>(
    (highest, severity) => {
      if (!highest) return severity
      return severityRank[severity] > severityRank[highest] ? severity : highest
    },
    null
  )
}

const hasAlertSummary = (summary: ScrapedItemAlertSummary | null | undefined): boolean =>
  Boolean(summary && Number(summary.total || 0) > 0)

const formatAlertSummaryLabel = (
  summary: ScrapedItemAlertSummary | null | undefined
): string => {
  const severity = getHighestSeverity(summary)
  const prefix = formatSeverityLabel(severity)
  return `${prefix} alert match`
}

const startOfTodayIso = (): string => {
  const now = new Date()
  now.setHours(0, 0, 0, 0)
  return now.toISOString()
}

const renderItemTimestamp = (item: ScrapedItem, t: TFunction) => {
  if (item.published_at) return formatRelativeTime(item.published_at, t)
  return formatRelativeTime(item.created_at, t)
}

const getItemPreviewText = (item: ScrapedItem): string | null => {
  const raw = item.summary || item.content || ""
  const text = stripHtmlToText(raw)
  if (!text) return null
  if (text.length <= 160) return text
  return `${text.slice(0, 159)}…`
}

const isLikelyHtml = (value: string): boolean => /<\/?[a-z][\s\S]*>/i.test(value)

const formatReaderDate = (value: string | null | undefined): string => {
  if (!value) return ""
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return ""
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(parsed)
}

const getDomain = (url: string | null | undefined): string | null => {
  if (!url) return null
  try {
    return new URL(url).hostname
  } catch {
    return null
  }
}

const isEditableTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tagName = target.tagName.toLowerCase()
  if (tagName === "input" || tagName === "textarea" || tagName === "select") {
    return true
  }
  if (target.closest("[contenteditable='true']")) return true
  if (target.closest(".ant-select-dropdown")) return true
  return false
}

const useSafeNavigate = () => {
  try {
    return useNavigate()
  } catch {
    return null
  }
}

export const ItemsTab: React.FC = () => {
  const navigate = useSafeNavigate()
  const { t } = useTranslation(["watchlists", "common"])
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const openSourceForm = useWatchlistsStore((s) => s.openSourceForm)
  const openJobForm = useWatchlistsStore((s) => s.openJobForm)
  const openRunDetail = useWatchlistsStore((s) => s.openRunDetail)
  const setRunsJobFilter = useWatchlistsStore((s) => s.setRunsJobFilter)
  const setRunsStatusFilter = useWatchlistsStore((s) => s.setRunsStatusFilter)
  const setOutputsJobFilter = useWatchlistsStore((s) => s.setOutputsJobFilter)
  const setOutputsRunFilter = useWatchlistsStore((s) => s.setOutputsRunFilter)
  const selectedRunId = useWatchlistsStore((s) => s.selectedRunId)
  const selectedSourceId = useWatchlistsStore((s) => s.itemsSelectedSourceId)
  const setStoreSelectedSourceId = useWatchlistsStore((s) => s.setItemsSelectedSourceId)
  const statusFilterState = useWatchlistsStore((s) => s.itemsStatusFilter)
  const setStoreStatusFilter = useWatchlistsStore((s) => s.setItemsStatusFilter)
  const smartFilterState = useWatchlistsStore((s) => s.itemsSmartFilter)
  const setStoreSmartFilter = useWatchlistsStore((s) => s.setItemsSmartFilter)
  const itemsSearch = useWatchlistsStore((s) => s.itemsSearchQuery)
  const setStoreItemsSearch = useWatchlistsStore((s) => s.setItemsSearchQuery)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)

  const [sources, setSources] = useState<WatchlistSource[]>([])
  const [sourcesLoading, setSourcesLoading] = useState(false)
  const [sourcesCappedAtLimit, setSourcesCappedAtLimit] = useState(false)
  const [sourceSearch, setSourceSearch] = useState("")
  const [runs, setRuns] = useState<WatchlistRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [queueRunFilter, setQueueRunFilter] = useState<number | null>(selectedRunId)
  const [visibleSourceCount, setVisibleSourceCount] = useState(SOURCE_LIST_INITIAL_RENDER_COUNT)
  const [items, setItems] = useState<ScrapedItem[]>([])
  const [itemsLoading, setItemsLoading] = useState(false)
  const [itemsTotal, setItemsTotal] = useState(0)
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null)
  const [itemsPage, setItemsPage] = useState(1)
  const [itemsPageSize, setItemsPageSize] = useState<number>(() =>
    loadPersistedItemPageSize(
      typeof window !== "undefined" ? window.localStorage : undefined
    )
  )
  const [sortMode, setSortMode] = useState<ReaderSortMode>(() =>
    loadPersistedItemsSortMode(
      typeof window !== "undefined" ? window.localStorage : undefined
    )
  )
  const [updatingItemId, setUpdatingItemId] = useState<number | null>(null)
  const [queueGenerating, setQueueGenerating] = useState(false)
  const [selectedItemIds, setSelectedItemIds] = useState<number[]>([])
  const [batchReviewScope, setBatchReviewScope] = useState<BatchReviewScope | null>(null)
  const [batchReviewProgress, setBatchReviewProgress] = useState<BatchReviewProgress | null>(null)
  const [collectingAllFiltered, setCollectingAllFiltered] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [shortcutsHintVisible, setShortcutsHintVisible] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    try {
      const raw = window.localStorage.getItem(SHORTCUTS_HINT_DISMISSED_STORAGE_KEY)
      return raw !== "1"
    } catch {
      return true
    }
  })
  const [itemsLiveAnnouncement, setItemsLiveAnnouncement] = useState("")
  const statusFilter = normalizeStatusFilter(statusFilterState)
  const smartFilter = normalizeSmartFilter(smartFilterState)
  const defaultViewPresets = useMemo(
    () =>
      buildDefaultItemsViewPresets({
        [SYSTEM_ITEMS_VIEW_PRESET_IDS.unreadToday]: t(
          "watchlists:items.savedViews.defaults.unreadToday",
          "Unread today"
        ),
        [SYSTEM_ITEMS_VIEW_PRESET_IDS.highPriority]: t(
          "watchlists:items.savedViews.defaults.highPriority",
          "High-priority"
        ),
        [SYSTEM_ITEMS_VIEW_PRESET_IDS.needsReview]: t(
          "watchlists:items.savedViews.defaults.needsReview",
          "Needs review"
        )
      }),
    [t]
  )
  const [viewPresets, setViewPresets] = useState<ItemsViewPreset[]>(() =>
    provisionItemsViewPresets(
      loadPersistedItemsViewPresets(
        typeof window !== "undefined" ? window.localStorage : undefined
      ),
      defaultViewPresets
    ).map((preset) => ({
      ...preset,
      smartFilter: normalizeSmartFilter(preset.smartFilter),
      statusFilter: normalizeStatusFilter(preset.statusFilter),
      sortMode: normalizeReaderSortMode(preset.sortMode)
    }))
  )
  const [viewPresetsLoading, setViewPresetsLoading] = useState(false)
  const [legacyViewPresets, setLegacyViewPresets] = useState<ItemsViewPreset[]>([])
  const [importingLocalViews, setImportingLocalViews] = useState(false)
  const [activePresetId, setActivePresetId] = useState<string | null>(null)
  const [saveViewModalOpen, setSaveViewModalOpen] = useState(false)
  const [newViewName, setNewViewName] = useState("")
  const [smartCounts, setSmartCounts] = useState<Record<SmartFeedFilter, number>>({
    all: 0,
    today: 0,
    todayUnread: 0,
    unread: 0,
    reviewed: 0,
    queued: 0,
    alertMatches: 0
  })

  const sourceNameById = useMemo(
    () => new Map<number, string>(sources.map((source) => [source.id, source.name])),
    [sources]
  )

  const selectedSourceName = useMemo(() => {
    if (selectedSourceId === null) {
      return t("watchlists:items.allFeeds", "All Feeds")
    }
    return (
      sourceNameById.get(selectedSourceId) ||
      t("watchlists:items.unknownSource", "Unknown source")
    )
  }, [selectedSourceId, sourceNameById, t])

  const queueRunOptions = useMemo(
    () =>
      runs.map((run) => ({
        value: run.id,
        label: t("watchlists:items.queue.runOption", "Run #{{id}}", {
          id: run.id
        })
      })),
    [runs, t]
  )

  const filteredSources = useMemo(
    () => filterSourcesForReader(sources, sourceSearch),
    [sources, sourceSearch]
  )
  const orderedSources = useMemo(
    () => orderSourcesForReader(filteredSources, selectedSourceId),
    [filteredSources, selectedSourceId]
  )
  const visibleSources = useMemo(
    () => orderedSources.slice(0, visibleSourceCount),
    [orderedSources, visibleSourceCount]
  )
  const hasCollapsedSources = visibleSources.length < orderedSources.length

  const sortedItems = items

  const selectedItem = useMemo(
    () => sortedItems.find((item) => item.id === selectedItemId) || null,
    [selectedItemId, sortedItems]
  )

  const selectedItemRawBody = selectedItem
    ? selectedItem.content || selectedItem.summary || ""
    : ""

  const selectedItemBodyIsHtml = isLikelyHtml(selectedItemRawBody)

  const selectedItemBodyHtml = useMemo(() => {
    if (!selectedItemRawBody) return ""
    return DOMPurify.sanitize(selectedItemRawBody, { USE_PROFILES: { html: true } })
  }, [selectedItemRawBody])

  const selectedItemPreviewImage = useMemo(
    () => extractImageUrl(selectedItem?.content || selectedItem?.summary || null),
    [selectedItem]
  )

  const itemImagesById = useMemo(() => {
    const map = new Map<number, string>()
    for (const item of items) {
      const image = extractImageUrl(item.content || item.summary)
      if (image) map.set(item.id, image)
    }
    return map
  }, [items])

  const itemPreviewById = useMemo(() => {
    const map = new Map<number, string>()
    for (const item of items) {
      const preview = getItemPreviewText(item)
      if (preview) map.set(item.id, preview)
    }
    return map
  }, [items])

  const currentUnreadCount = useMemo(
    () => smartCounts.unread,
    [smartCounts.unread]
  )
  const requiresReviewedRefresh = shouldReloadItemsAfterReviewMutation(smartFilter)

  const searchQuery = itemsSearch.trim()
  const [effectiveSearchQuery, setEffectiveSearchQuery] = useState(searchQuery)
  const sourcesRequestTokenRef = useRef(0)
  const itemsRequestTokenRef = useRef(0)
  const smartCountsRequestTokenRef = useRef(0)
  const smartCountsCacheRef = useRef<Record<string, SmartCountsCacheEntry>>({})
  const sourceListRef = useRef<HTMLDivElement | null>(null)
  const shortcutsTriggerRef = useRef<HTMLElement | null>(null)
  const shortcutsHelpButtonRef = useRef<HTMLButtonElement | null>(null)
  const saveViewTriggerRef = useRef<HTMLElement | null>(null)
  const wasShortcutsOpenRef = useRef(false)
  const wasSaveViewModalOpenRef = useRef(false)
  const hasItemsAnnouncementBaselineRef = useRef(false)
  const previousSelectedItemIdRef = useRef<number | null>(null)
  const skippedInitialAutoSelectionRef = useRef(false)

  useEffect(() => {
    const timeout = setTimeout(() => {
      setEffectiveSearchQuery(searchQuery)
    }, 180)
    return () => clearTimeout(timeout)
  }, [searchQuery])

  useEffect(() => {
    const currentSelectedId = selectedItem?.id ?? null

    if (!hasItemsAnnouncementBaselineRef.current) {
      hasItemsAnnouncementBaselineRef.current = true
      previousSelectedItemIdRef.current = currentSelectedId
      return
    }

    if (previousSelectedItemIdRef.current === currentSelectedId) return

    if (
      !skippedInitialAutoSelectionRef.current &&
      previousSelectedItemIdRef.current == null &&
      currentSelectedId != null
    ) {
      skippedInitialAutoSelectionRef.current = true
      previousSelectedItemIdRef.current = currentSelectedId
      return
    }

    if (!selectedItem) {
      setItemsLiveAnnouncement(
        t("watchlists:items.live.selectionCleared", "No update selected.")
      )
    } else {
      const title = selectedItem.title || t("watchlists:items.untitled", "Untitled item")
      const source =
        sourceNameById.get(selectedItem.source_id) ||
        t("watchlists:items.unknownSource", "Unknown source")
      const reviewState = selectedItem.reviewed
        ? t("watchlists:items.rowStatusReviewed", "Reviewed")
        : t("watchlists:items.rowStatusUnread", "Unread")
      setItemsLiveAnnouncement(
        t(
          "watchlists:items.live.selectionChanged",
          "Selected {{title}} from {{source}} ({{state}}).",
          {
            title,
            source,
            state: reviewState
          }
        )
      )
    }

    previousSelectedItemIdRef.current = currentSelectedId
  }, [selectedItem, sourceNameById, t])

  const buildBaseFilterParams = useCallback(
    (
      overrides: Partial<FetchItemsParams> = {},
      queryOverride: string = effectiveSearchQuery
    ): FetchItemsParams => {
      const params: FetchItemsParams = {
        watchlist_id: selectedWatchlistId ?? undefined,
        source_id: selectedSourceId ?? undefined,
        status: statusFilter === "all" ? undefined : statusFilter,
        sort: toServerItemSortMode(sortMode),
        include_alert_summary: true,
        q: queryOverride || undefined,
        ...overrides
      }

      if (smartFilter === "today") {
        params.since = startOfTodayIso()
      } else if (smartFilter === "todayUnread") {
        params.since = startOfTodayIso()
        params.reviewed = false
      } else if (smartFilter === "unread") {
        params.reviewed = false
      } else if (smartFilter === "reviewed") {
        params.reviewed = true
      } else if (smartFilter === "queued") {
        params.queued_for_briefing = true
        params.run_id = queueRunFilter ?? undefined
      } else if (smartFilter === "alertMatches") {
        params.has_alert = true
      }

      return params
    },
    [effectiveSearchQuery, queueRunFilter, selectedSourceId, selectedWatchlistId, smartFilter, sortMode, statusFilter]
  )

  const loadSources = useCallback(async () => {
    const requestToken = sourcesRequestTokenRef.current + 1
    sourcesRequestTokenRef.current = requestToken
    setSourcesLoading(true)
    try {
      const loaded: WatchlistSource[] = []
      let sourceTotal = 0
      let page = 1

      while (loaded.length < SOURCE_LOAD_MAX_ITEMS) {
        const response = await fetchWatchlistSources({
          watchlist_id: selectedWatchlistId ?? undefined,
          page,
          size: SOURCE_LOAD_PAGE_SIZE
        })
        if (requestToken !== sourcesRequestTokenRef.current) return
        const batch = Array.isArray(response.items) ? response.items : []
        if (page === 1) {
          sourceTotal = Number(response.total || batch.length)
        }
        loaded.push(...batch)
        if (
          batch.length < SOURCE_LOAD_PAGE_SIZE ||
          loaded.length >= (response.total || 0)
        ) {
          break
        }
        page += 1
      }

      if (requestToken !== sourcesRequestTokenRef.current) return
      const capped = loaded.slice(0, SOURCE_LOAD_MAX_ITEMS)
      setSources(capped)
      const normalizedTotal = sourceTotal > 0 ? sourceTotal : capped.length
      setSourcesCappedAtLimit(normalizedTotal > SOURCE_LOAD_MAX_ITEMS)
    } catch (error) {
      if (requestToken !== sourcesRequestTokenRef.current) return
      console.error("Failed to load watchlist sources:", error)
      message.error(t("watchlists:items.sourcesError", "Failed to load sources"))
      setSourcesCappedAtLimit(false)
    } finally {
      if (requestToken !== sourcesRequestTokenRef.current) return
      setSourcesLoading(false)
    }
  }, [selectedWatchlistId, t])

  const loadRuns = useCallback(async () => {
    setRunsLoading(true)
    try {
      const response = await fetchWatchlistRuns({
        watchlist_id: selectedWatchlistId ?? undefined,
        page: 1,
        size: 200
      })
      setRuns(Array.isArray(response.items) ? response.items : [])
    } catch (error) {
      console.error("Failed to load watchlist runs:", error)
      setRuns([])
    } finally {
      setRunsLoading(false)
    }
  }, [selectedWatchlistId])

  const expandVisibleSourcesIfNeeded = useCallback(() => {
    const listElement = sourceListRef.current
    if (!listElement) return
    if (!hasCollapsedSources) return
    if (
      !shouldExpandSourceRenderWindow(
        listElement.scrollTop,
        listElement.scrollHeight,
        listElement.clientHeight
      )
    ) {
      return
    }
    setVisibleSourceCount((currentCount) =>
      getNextSourceRenderCount(currentCount, orderedSources.length)
    )
  }, [hasCollapsedSources, orderedSources.length])

  const loadItems = useCallback(async ({ silent = false }: { silent?: boolean } = {}) => {
    const requestToken = itemsRequestTokenRef.current + 1
    itemsRequestTokenRef.current = requestToken
    if (!silent) setItemsLoading(true)
    try {
      const response = await fetchScrapedItems(
        buildBaseFilterParams({
          page: itemsPage,
          size: itemsPageSize
        })
      )
      if (requestToken !== itemsRequestTokenRef.current) return
      const nextItems = Array.isArray(response.items) ? response.items : []
      setItems(nextItems)
      setItemsTotal(response.total || nextItems.length)
      setSelectedItemId((prev) => resolveSelectedItemId(prev, nextItems))
    } catch (error) {
      if (requestToken !== itemsRequestTokenRef.current) return
      console.error("Failed to load watchlist items:", error)
      if (!silent) {
        message.error(t("watchlists:items.fetchError", "Failed to load updates"))
        setItems([])
        setItemsTotal(0)
        setSelectedItemId(null)
      }
    } finally {
      if (requestToken !== itemsRequestTokenRef.current) return
      if (!silent) setItemsLoading(false)
    }
  }, [buildBaseFilterParams, itemsPage, itemsPageSize, t])

  const loadSmartCounts = useCallback(async () => {
    const requestToken = smartCountsRequestTokenRef.current + 1
    smartCountsRequestTokenRef.current = requestToken
    const cacheKey = [
      selectedWatchlistId ?? "all-watchlists",
      selectedSourceId ?? "all",
      statusFilter,
      queueRunFilter ?? "all-runs",
      effectiveSearchQuery.trim().toLowerCase()
    ].join("|")
    const cached = smartCountsCacheRef.current[cacheKey]
    if (cached && Date.now() - cached.cachedAt < SMART_COUNTS_CACHE_TTL_MS) {
      setSmartCounts(cached.counts)
      return
    }
    try {
      const base = {
        watchlist_id: selectedWatchlistId ?? undefined,
        source_id: selectedSourceId ?? undefined,
        status: statusFilter === "all" ? undefined : statusFilter,
        q: effectiveSearchQuery || undefined,
        queue_run_id: queueRunFilter ?? undefined
      }
      const [counts, alertCounts] = await Promise.all([
        fetchScrapedItemSmartCounts(base),
        fetchScrapedItemSmartCounts({
          ...base,
          has_alert: true
        })
      ])

      if (requestToken !== smartCountsRequestTokenRef.current) return
      const nextCounts = {
        all: Number(counts.all || 0),
        today: Number(counts.today || 0),
        todayUnread: Number(counts.today_unread || 0),
        unread: Number(counts.unread || 0),
        reviewed: Number(counts.reviewed || 0),
        queued: Number(counts.queued || 0),
        alertMatches: Number(alertCounts.all || 0)
      }
      setSmartCounts(nextCounts)
      smartCountsCacheRef.current[cacheKey] = {
        counts: nextCounts,
        cachedAt: Date.now()
      }
    } catch (error) {
      if (requestToken !== smartCountsRequestTokenRef.current) return
      console.error("Failed to load smart feed counts:", error)
    }
  }, [effectiveSearchQuery, queueRunFilter, selectedSourceId, selectedWatchlistId, statusFilter])

  const invalidateSmartCountsCache = useCallback(() => {
    smartCountsCacheRef.current = {}
  }, [])

  const refreshItemsView = useCallback(() => {
    void loadSources()
    void loadItems()
    void loadSmartCounts()
  }, [loadItems, loadSmartCounts, loadSources])

  useEffect(() => {
    void loadSources()
  }, [loadSources])

  useEffect(() => {
    void loadRuns()
  }, [loadRuns])

  useEffect(() => {
    void loadItems()
  }, [loadItems])

  // Auto-refresh items every 30 seconds to pick up new content from runs
  useEffect(() => {
    let pollInFlight = false
    const interval = setInterval(() => {
      if (pollInFlight || document.visibilityState !== "visible") return
      pollInFlight = true
      void Promise.all([
        loadItems({ silent: true }),
        loadSmartCounts()
      ]).finally(() => {
        pollInFlight = false
      })
    }, 30_000)
    return () => {
      clearInterval(interval)
      pollInFlight = false
    }
  }, [loadItems, loadSmartCounts])

  useEffect(() => {
    void loadSmartCounts()
  }, [loadSmartCounts])

  useEffect(() => {
    setVisibleSourceCount((currentCount) => {
      const nextInitial = getInitialSourceRenderCount(orderedSources.length, sourceSearch)
      if (nextInitial === 0) return 0
      if (sourceSearch.trim().length > 0) {
        return orderedSources.length
      }
      if (currentCount === 0) return nextInitial
      return Math.min(currentCount, orderedSources.length)
    })
  }, [orderedSources.length, sourceSearch])

  useEffect(() => {
    expandVisibleSourcesIfNeeded()
  }, [expandVisibleSourcesIfNeeded, visibleSourceCount])

  useEffect(() => {
    persistItemPageSize(
      typeof window !== "undefined" ? window.localStorage : undefined,
      itemsPageSize
    )
  }, [itemsPageSize])

  useEffect(() => {
    if (selectedWatchlistId != null) return
    persistItemsViewPresets(
      typeof window !== "undefined" ? window.localStorage : undefined,
      viewPresets
    )
  }, [selectedWatchlistId, viewPresets])

  useEffect(() => {
    persistItemsSortMode(
      typeof window !== "undefined" ? window.localStorage : undefined,
      sortMode
    )
  }, [sortMode])

  useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(
        SHORTCUTS_HINT_DISMISSED_STORAGE_KEY,
        shortcutsHintVisible ? "0" : "1"
      )
    } catch {
      // Ignore storage write errors (private browsing, quota, etc.)
    }
  }, [shortcutsHintVisible])

  useEffect(() => {
    if (shortcutsOpen) {
      wasShortcutsOpenRef.current = true
      return
    }

    if (wasShortcutsOpenRef.current) {
      wasShortcutsOpenRef.current = false
      restoreFocusToElement(shortcutsTriggerRef.current)
    }
  }, [shortcutsOpen])

  useEffect(() => {
    if (saveViewModalOpen) {
      wasSaveViewModalOpenRef.current = true
      return
    }

    if (wasSaveViewModalOpenRef.current) {
      wasSaveViewModalOpenRef.current = false
      restoreFocusToElement(saveViewTriggerRef.current)
    }
  }, [saveViewModalOpen])

  useEffect(() => {
    const visibleIds = new Set(sortedItems.map((item) => item.id))
    setSelectedItemIds((prev) => {
      if (prev.length === 0) return prev
      const filtered = prev.filter((id) => visibleIds.has(id))
      return filtered.length === prev.length ? prev : filtered
    })
  }, [sortedItems])

  useEffect(() => {
    setSelectedItemId((prev) => resolveSelectedItemId(prev, sortedItems))
  }, [sortedItems])

  useEffect(() => {
    setSelectedItemIds((prev) => (prev.length === 0 ? prev : []))
  }, [selectedSourceId, smartFilter, statusFilter, searchQuery, itemsPage])

  useEffect(() => {
    if (selectedRunId && queueRunFilter == null) {
      setQueueRunFilter(selectedRunId)
    }
  }, [queueRunFilter, selectedRunId])

  useEffect(() => {
    if (smartFilter !== "queued") return
    if (queueRunFilter != null) return
    const firstRunId = runs[0]?.id
    if (typeof firstRunId === "number") {
      setQueueRunFilter(firstRunId)
    }
  }, [queueRunFilter, runs, smartFilter])

  useEffect(() => {
    const matchingPreset = viewPresets.find(
      (preset) =>
        preset.sourceId === selectedSourceId &&
        preset.smartFilter === smartFilter &&
        preset.statusFilter === statusFilter &&
        preset.sortMode === sortMode &&
        preset.searchQuery === searchQuery
    )
    const nextId = matchingPreset?.id ?? null
    setActivePresetId((prev) => (prev === nextId ? prev : nextId))
  }, [searchQuery, selectedSourceId, smartFilter, sortMode, statusFilter, viewPresets])

  const handleSourceSelect = (sourceId: number | null) => {
    setStoreSelectedSourceId(sourceId)
    setItemsPage(1)
  }

  const handleStatusChange = (nextStatus: ReaderStatusFilter) => {
    setStoreStatusFilter(nextStatus)
    setItemsPage(1)
  }

  const handleSmartFilterChange = (nextFilter: SmartFeedFilter) => {
    setStoreSmartFilter(nextFilter)
    setItemsPage(1)
  }

  const handleQueueRunFilterChange = (runId: number | null) => {
    setQueueRunFilter(runId)
    setItemsPage(1)
  }

  const handleToggleReviewed = useCallback(async (item: ScrapedItem) => {
    setUpdatingItemId(item.id)
    try {
      const updated = await updateScrapedItem(item.id, { reviewed: !item.reviewed })
      setItems((prev) =>
        prev.map((entry) => (entry.id === updated.id ? updated : entry))
      )
      message.success(
        updated.reviewed
          ? t("watchlists:items.markedReviewed", "Marked as reviewed")
          : t("watchlists:items.markedUnreviewed", "Marked as unreviewed")
      )
      if (requiresReviewedRefresh) {
        void loadItems()
      }
      invalidateSmartCountsCache()
      void loadSmartCounts()
    } catch (error) {
      console.error("Failed to update watchlist item:", error)
      message.error(t("watchlists:items.updateError", "Failed to update item"))
    } finally {
      setUpdatingItemId(null)
    }
  }, [invalidateSmartCountsCache, loadItems, loadSmartCounts, requiresReviewedRefresh, t])

  const handleToggleBriefingQueue = useCallback(async (item: ScrapedItem) => {
    const isQueued = Boolean(item.queued_for_briefing)
    setUpdatingItemId(item.id)
    try {
      const updated = await updateScrapedItem(item.id, {
        queued_for_briefing: !isQueued
      })
      setItems((prev) =>
        prev.map((entry) => (entry.id === updated.id ? updated : entry))
      )
      message.success(
        !isQueued
          ? t(
              "watchlists:items.briefingIncluded",
              "Added to the next briefing queue."
            )
          : t(
              "watchlists:items.briefingRemoved",
              "Removed from the briefing queue."
            )
      )
      if (smartFilter === "queued") {
        await loadItems()
      }
      invalidateSmartCountsCache()
      await loadSmartCounts()
    } catch (error) {
      console.error("Failed to update watchlist item briefing queue:", error)
      message.error(
        t(
          "watchlists:items.briefingIncludeError",
          "Failed to include item in briefing."
        )
      )
    } finally {
      setUpdatingItemId(null)
    }
  }, [loadItems, loadSmartCounts, smartFilter, t])

  const handleGenerateReportFromQueue = useCallback(async () => {
    const runId = queueRunFilter
    if (!runId) {
      message.info(
        t(
          "watchlists:items.queue.runRequired",
          "Select a run to generate a queued report."
        )
      )
      return
    }

    setQueueGenerating(true)
    try {
      const queuePageSize = 200
      const queuedItems: ScrapedItem[] = []
      let page = 1

      while (true) {
        const response = await fetchScrapedItems({
          run_id: runId,
          queued_for_briefing: true,
          page,
          size: queuePageSize
        })
        const batch = Array.isArray(response.items) ? response.items : []
        queuedItems.push(...batch)
        if (batch.length < queuePageSize || queuedItems.length >= Number(response.total || 0)) {
          break
        }
        page += 1
      }

      if (queuedItems.length === 0) {
        message.info(
          t(
            "watchlists:items.queue.empty",
            "No queued items found for this run."
          )
        )
        return
      }

      await createWatchlistOutput({
        run_id: runId,
        item_ids: queuedItems.map((item) => item.id)
      })
      message.success(
        t(
          "watchlists:items.queue.generated",
          "Created report from {{count}} queued item{{plural}}.",
          {
            count: queuedItems.length,
            plural: queuedItems.length === 1 ? "" : "s"
          }
        )
      )

      setOutputsRunFilter(runId)
      const jobId = queuedItems[0]?.job_id
      if (typeof jobId === "number") {
        setOutputsJobFilter(jobId)
      }
      setActiveTab("outputs")
    } catch (error) {
      console.error("Failed to generate queued watchlist report:", error)
      message.error(
        t(
          "watchlists:items.queue.generateError",
          "Failed to generate report from queued items."
        )
      )
    } finally {
      setQueueGenerating(false)
    }
  }, [queueRunFilter, setActiveTab, setOutputsJobFilter, setOutputsRunFilter, t])

  const pageItemIds = useMemo(() => sortedItems.map((item) => item.id), [sortedItems])

  const pageUnreviewedItemIds = useMemo(
    () => sortedItems.filter((item) => !item.reviewed).map((item) => item.id),
    [sortedItems]
  )

  const selectedItemIdSet = useMemo(
    () => new Set(selectedItemIds),
    [selectedItemIds]
  )

  const selectedUnreviewedItemIds = useMemo(
    () =>
      sortedItems
        .filter((item) => selectedItemIdSet.has(item.id) && !item.reviewed)
        .map((item) => item.id),
    [selectedItemIdSet, sortedItems]
  )
  const selectedUnreviewedCount = selectedUnreviewedItemIds.length
  const pageUnreviewedCount = pageUnreviewedItemIds.length
  const allFilteredUnreadEstimate = useMemo(() => {
    if (smartFilter === "reviewed") return 0
    if (smartFilter === "today" || smartFilter === "todayUnread") return smartCounts.todayUnread
    if (smartFilter === "queued") return smartCounts.queued
    if (smartFilter === "alertMatches") return smartCounts.alertMatches
    return smartCounts.unread
  }, [smartCounts, smartFilter])

  const allPageItemsSelected = useMemo(
    () => pageItemIds.length > 0 && pageItemIds.every((id) => selectedItemIdSet.has(id)),
    [pageItemIds, selectedItemIdSet]
  )

  const somePageItemsSelected = useMemo(
    () => !allPageItemsSelected && pageItemIds.some((id) => selectedItemIdSet.has(id)),
    [allPageItemsSelected, pageItemIds, selectedItemIdSet]
  )

  const pageSizeOptions = useMemo(
    () =>
      ITEM_PAGE_SIZE_OPTIONS.map((value) => ({
        label: t("common:pagination.itemsPerPage", "{{count}} / page", { count: value }),
        value
      })),
    [t]
  )

  const getBatchScopeLabel = useCallback(
    (scope: BatchReviewScope, count: number) => {
      if (scope === "selected") {
        return t("watchlists:items.batch.scope.selected", "selected item{{plural}}", {
          plural: count === 1 ? "" : "s"
        })
      }
      if (scope === "page") {
        return t("watchlists:items.batch.scope.page", "updates on this page")
      }
      return t("watchlists:items.batch.scope.allFiltered", "all filtered updates")
    },
    [t]
  )

  const viewPresetOptions = useMemo(
    () =>
      viewPresets.map((preset) => ({
        label: preset.name,
        value: preset.id
      })),
    [viewPresets]
  )
  const normalizeViewPresets = useCallback(
    (presets: PersistedItemsViewPreset[]): ItemsViewPreset[] =>
      provisionItemsViewPresets(presets, defaultViewPresets).map((preset) => ({
        ...preset,
        smartFilter: normalizeSmartFilter(preset.smartFilter),
        statusFilter: normalizeStatusFilter(preset.statusFilter),
        sortMode: normalizeReaderSortMode(preset.sortMode)
      })),
    [defaultViewPresets]
  )

  const buildCurrentServerViewPayload = useCallback(
    (name: string) =>
      buildServerItemViewCreatePayload(
        makeCurrentViewPreset(
          name,
          selectedSourceId,
          smartFilter,
          statusFilter,
          sortMode,
          searchQuery
        )
      ),
    [searchQuery, selectedSourceId, smartFilter, sortMode, statusFilter]
  )

  useEffect(() => {
    let cancelled = false

    const loadServerSavedViews = async () => {
      const storage = typeof window !== "undefined" ? window.localStorage : undefined
      const migratable = getMigratableItemsViewPresets(storage).map((preset) => ({
        ...preset,
        smartFilter: normalizeSmartFilter(preset.smartFilter),
        statusFilter: normalizeStatusFilter(preset.statusFilter),
        sortMode: normalizeReaderSortMode(preset.sortMode)
      }))

      if (selectedWatchlistId == null) {
        setLegacyViewPresets([])
        setViewPresets(
          normalizeViewPresets(loadPersistedItemsViewPresets(storage))
        )
        return
      }

      setViewPresetsLoading(true)
      try {
        const result = await fetchWatchlistItemViews(selectedWatchlistId)
        if (cancelled) return
        const serverPresets = (Array.isArray(result.items) ? result.items : [])
          .map(toItemsViewPresetFromServer)
        setLegacyViewPresets(migratable)
        setViewPresets(normalizeViewPresets(serverPresets))
      } catch (error) {
        if (cancelled) return
        console.error("Failed to load watchlist item saved views:", error)
        setLegacyViewPresets(migratable)
        setViewPresets(normalizeViewPresets(loadPersistedItemsViewPresets(storage)))
        message.error(
          t("watchlists:items.savedViews.loadError", "Failed to load saved views.")
        )
      } finally {
        if (!cancelled) setViewPresetsLoading(false)
      }
    }

    void loadServerSavedViews()
    return () => {
      cancelled = true
    }
  }, [normalizeViewPresets, selectedWatchlistId, t])

  const activePresetIsSystem = useMemo(
    () => isSystemItemsViewPresetId(activePresetId),
    [activePresetId]
  )

  const handleToggleItemSelected = useCallback((itemId: number, checked: boolean) => {
    setSelectedItemIds((prev) => {
      if (checked) return Array.from(new Set([...prev, itemId]))
      return prev.filter((id) => id !== itemId)
    })
  }, [])

  const handleToggleSelectPage = useCallback((checked: boolean) => {
    setSelectedItemIds((prev) => {
      const next = new Set(prev)
      if (checked) {
        pageItemIds.forEach((id) => next.add(id))
      } else {
        pageItemIds.forEach((id) => next.delete(id))
      }
      return Array.from(next)
    })
  }, [pageItemIds])

  const buildAllFilteredBatchScope = useCallback((): ScrapedItemBatchScope | null => {
    if (smartFilter === "reviewed") {
      return null
    }
    const scope = { ...buildBaseFilterParams({}, searchQuery), reviewed: false }
    delete scope.include_alert_summary
    delete scope.page
    delete scope.size
    delete scope.sort
    return scope
  }, [buildBaseFilterParams, searchQuery, smartFilter])

  const countAllFilteredBatchScope = useCallback(async (
    scope: ScrapedItemBatchScope
  ): Promise<number> => {
    const response = await fetchScrapedItems({
      ...scope,
      page: 1,
      size: 1
    })
    return Number(response.total || 0)
  }, [])

  const applyBatchReviewResult = useCallback((
    result: ScrapedItemBatchUpdateResponse,
    fallbackIds: number[]
  ) => {
    const changedIds = Array.isArray(result.changed_ids)
      ? result.changed_ids
      : fallbackIds
    if (changedIds.length === 0) return
    const changedIdSet = new Set(changedIds)
    setItems((prev) =>
      prev.map((item) =>
        changedIdSet.has(item.id)
          ? { ...item, reviewed: true }
          : item
      )
    )
    setSelectedItemIds((prev) => prev.filter((id) => !changedIdSet.has(id)))
  }, [])

  const markItemsReviewed = useCallback(async (
    itemIds: number[] | null,
    scope: BatchReviewScope,
    estimatedCount?: number,
    batchScopeParams?: ScrapedItemBatchScope | null
  ): Promise<void> => {
    if (selectedWatchlistId == null) {
      message.error(t("watchlists:items.batch.watchlistRequired", "Select a Watchlist first."))
      return
    }

    const uniqueIds = Array.from(new Set(itemIds || []))
    const resolvedBatchScope = itemIds ? null : (batchScopeParams ?? buildAllFilteredBatchScope())
    const totalEstimate = itemIds ? uniqueIds.length : Number(estimatedCount || 0)
    if (totalEstimate === 0) return
    if (!itemIds && !resolvedBatchScope) {
      message.info(
        t(
          "watchlists:items.batch.noFiltered",
          "No matching unread updates to review."
        )
      )
      return
    }

    setBatchReviewScope(scope)
    setBatchReviewProgress({
      scope,
      phase: "running",
      isRunning: true,
      total: totalEstimate,
      processed: 0,
      succeeded: 0,
      failed: 0,
      incomplete: false,
      failedIds: [],
      failedItemIds: []
    })
    try {
      const result = await batchUpdateScrapedItems(
        itemIds
          ? {
              watchlist_id: selectedWatchlistId,
              item_ids: uniqueIds,
              reviewed: true
            }
          : {
              watchlist_id: selectedWatchlistId,
              scope: resolvedBatchScope ?? undefined,
              reviewed: true
            }
      )

      applyBatchReviewResult(result, uniqueIds)

      const cappedOrIncomplete = result.capped === true || result.exhausted === false
      const finalPhase: BatchReviewPhase =
        cappedOrIncomplete
          ? "partial"
          : Number(result.failed || 0) === 0
          ? "complete"
          : Number(result.changed || 0) > 0
            ? "partial"
            : "failed"
      const matched = Number(result.matched || totalEstimate)
      const changed = Number(result.changed || 0)
      const failed = Number(result.failed || 0)
      const failedItemIds = Array.isArray(result.failed_ids) ? result.failed_ids : []
      setBatchReviewProgress((previous) =>
        previous
          ? {
              ...previous,
              phase: finalPhase,
              isRunning: false,
              total: matched,
              processed: matched,
              succeeded: changed,
              failed,
              incomplete: cappedOrIncomplete,
              failedIds: [...failedItemIds],
              failedItemIds
            }
          : previous
      )

      if (changed > 0 && failed === 0 && !cappedOrIncomplete) {
        const scopeLabel = getBatchScopeLabel(scope, changed)
        message.success(
          t("watchlists:items.batch.completedScoped", "Marked {{count}} {{scope}} as reviewed.", {
            count: changed,
            scope: scopeLabel
          })
        )
      } else if (changed > 0 && cappedOrIncomplete) {
        const scopeLabel = getBatchScopeLabel(scope, changed)
        message.warning(
          t(
            "watchlists:items.batch.incompleteScoped",
            "Marked {{success}} {{scope}} as reviewed; more matching updates may remain.",
            {
              success: changed,
              scope: scopeLabel
            }
          )
        )
      } else if (changed > 0) {
        const scopeLabel = getBatchScopeLabel(scope, changed)
        message.warning(
          t(
            "watchlists:items.batch.partialScoped",
            "Marked {{success}} {{scope}} as reviewed; {{failed}} failed.",
            {
              success: changed,
              scope: scopeLabel,
              failed
            }
          )
        )
      } else {
        const scopeLabel = getBatchScopeLabel(scope, matched)
        message.error(
          t("watchlists:items.batch.failedScoped", "Failed to mark {{scope}} as reviewed.", {
            scope: scopeLabel
          })
        )
      }

      if (requiresReviewedRefresh) {
        invalidateSmartCountsCache()
        await Promise.all([loadItems(), loadSmartCounts()])
      } else {
        invalidateSmartCountsCache()
        await loadSmartCounts()
      }
    } catch (error) {
      console.error("Failed to apply batch reviewed update:", error)
      const scopeLabel = getBatchScopeLabel(scope, totalEstimate)
      setBatchReviewProgress((previous) =>
        previous
          ? {
              ...previous,
              phase: "failed",
              isRunning: false
            }
          : previous
      )
      message.error(
        t("watchlists:items.batch.failedScoped", "Failed to mark {{scope}} as reviewed.", {
          scope: scopeLabel
        })
      )
    } finally {
      setBatchReviewScope(null)
    }
  }, [
    applyBatchReviewResult,
    buildAllFilteredBatchScope,
    getBatchScopeLabel,
    invalidateSmartCountsCache,
    loadItems,
    loadSmartCounts,
    requiresReviewedRefresh,
    selectedWatchlistId,
    t
  ])

  const openBatchConfirm = useCallback((
    scope: BatchReviewScope,
    itemIds: number[] | null,
    count: number,
    title: string,
    batchScopeParams?: ScrapedItemBatchScope | null
  ) => {
    if (count === 0) return
    const scopeLabel = getBatchScopeLabel(scope, count)
    Modal.confirm({
      title,
      content: t(
        "watchlists:items.batch.confirmDescriptionScoped",
        "Scope: {{scope}}. This will mark {{count}} update{{plural}} as reviewed.",
        {
          scope: scopeLabel,
          count,
          plural: count === 1 ? "" : "s"
        }
      ),
      okText: t("watchlists:items.markReviewed", "Mark as reviewed"),
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => markItemsReviewed(itemIds, scope, count, batchScopeParams)
    })
  }, [getBatchScopeLabel, markItemsReviewed, t])

  const handleMarkSelectedReviewed = useCallback(() => {
    if (selectedUnreviewedItemIds.length === 0) {
      message.info(
        t("watchlists:items.batch.noSelected", "Select one or more items to review.")
      )
      return
    }

    openBatchConfirm(
      "selected",
      selectedUnreviewedItemIds,
      selectedUnreviewedItemIds.length,
      t("watchlists:items.batch.confirmSelectedTitle", "Mark selected updates as reviewed?")
    )
  }, [openBatchConfirm, selectedUnreviewedItemIds, t])

  const handleMarkPageReviewed = useCallback(() => {
    if (pageUnreviewedItemIds.length === 0) {
      message.info(
        t("watchlists:items.batch.noPage", "All updates on this page are already reviewed.")
      )
      return
    }

    openBatchConfirm(
      "page",
      pageUnreviewedItemIds,
      pageUnreviewedItemIds.length,
      t("watchlists:items.batch.confirmPageTitle", "Mark this page of updates as reviewed?")
    )
  }, [openBatchConfirm, pageUnreviewedItemIds, t])

  const handleMarkAllFilteredReviewed = useCallback(async () => {
    const batchScopeParams = buildAllFilteredBatchScope()
    if (!batchScopeParams) {
      message.info(
        t(
          "watchlists:items.batch.noFiltered",
          "No matching unread updates to review."
        )
      )
      return
    }

    setCollectingAllFiltered(true)
    try {
      const totalEstimate = await countAllFilteredBatchScope(batchScopeParams)
      if (totalEstimate === 0) {
        message.info(
          t(
            "watchlists:items.batch.noFiltered",
            "No matching unread updates to review."
          )
        )
        return
      }

      openBatchConfirm(
        "allFiltered",
        null,
        totalEstimate,
        t("watchlists:items.batch.confirmAllFilteredTitle", "Mark all filtered updates as reviewed?"),
        batchScopeParams
      )
    } catch (error) {
      console.error("Failed to count filtered items for batch review:", error)
      message.error(t("watchlists:items.fetchError", "Failed to load updates"))
    } finally {
      setCollectingAllFiltered(false)
    }
  }, [buildAllFilteredBatchScope, countAllFilteredBatchScope, openBatchConfirm, t])

  const batchProgressPercent = useMemo(() => {
    if (!batchReviewProgress) return 0
    if (batchReviewProgress.total <= 0) return 0
    return Math.min(100, Math.round((batchReviewProgress.processed / batchReviewProgress.total) * 100))
  }, [batchReviewProgress])

  const batchProgressSummary = useMemo(() => {
    if (!batchReviewProgress) return null
    if (batchReviewProgress.phase === "running") {
      return t(
        "watchlists:items.batch.progressRunning",
        "Processing {{processed}} of {{total}} items ({{succeeded}} succeeded, {{failed}} failed).",
        {
          processed: batchReviewProgress.processed,
          total: batchReviewProgress.total,
          succeeded: batchReviewProgress.succeeded,
          failed: batchReviewProgress.failed
        }
      )
    }
    if (batchReviewProgress.phase === "complete") {
      return t(
        "watchlists:items.batch.progressComplete",
        "Batch review complete: {{succeeded}} succeeded, {{failed}} failed.",
        {
          succeeded: batchReviewProgress.succeeded,
          failed: batchReviewProgress.failed
        }
      )
    }
    if (batchReviewProgress.phase === "partial") {
      if (batchReviewProgress.incomplete) {
        return t(
          "watchlists:items.batch.progressIncomplete",
          "Batch review partial: {{succeeded}} succeeded; more matching updates may remain.",
          {
            succeeded: batchReviewProgress.succeeded
          }
        )
      }
      return t(
        "watchlists:items.batch.progressPartial",
        "Batch review complete: {{succeeded}} succeeded, {{failed}} failed.",
        {
          succeeded: batchReviewProgress.succeeded,
          failed: batchReviewProgress.failed
        }
      )
    }
    return t(
      "watchlists:items.batch.progressFailed",
      "Batch review failed: {{failed}} failed.",
      {
        failed: batchReviewProgress.failed
      }
    )
  }, [batchReviewProgress, t])

  const retryFailedBatchReview = useCallback(() => {
    if (!batchReviewProgress) return
    if (batchReviewScope !== null) return
    if (batchReviewProgress.failedItemIds.length === 0) return
    void markItemsReviewed(
      batchReviewProgress.failedItemIds,
      batchReviewProgress.scope,
      batchReviewProgress.failedItemIds.length
    )
  }, [batchReviewProgress, batchReviewScope, markItemsReviewed])

  const handleSortModeChange = useCallback((nextSortMode: string) => {
    setSortMode(normalizeReaderSortMode(nextSortMode))
    setItemsPage(1)
  }, [])

  const applyViewPreset = useCallback((presetId: string) => {
    const preset = viewPresets.find((candidate) => candidate.id === presetId)
    if (!preset) return
    setStoreSelectedSourceId(preset.sourceId)
    setStoreSmartFilter(preset.smartFilter)
    setStoreStatusFilter(preset.statusFilter)
    setSortMode(normalizeReaderSortMode(preset.sortMode))
    setStoreItemsSearch(preset.searchQuery)
    setItemsPage(1)
    setActivePresetId(preset.id)
  }, [
    setStoreItemsSearch,
    setStoreSelectedSourceId,
    setStoreSmartFilter,
    setStoreStatusFilter,
    setSortMode,
    viewPresets
  ])

  const saveCurrentView = useCallback(async () => {
    if (activePresetId && !isSystemItemsViewPresetId(activePresetId)) {
      const preset = viewPresets.find((candidate) => candidate.id === activePresetId)
      if (!preset) return
      const payload = buildCurrentServerViewPayload(preset.name)

      if (selectedWatchlistId != null && typeof preset.serverId === "number") {
        try {
          const updated = await updateWatchlistItemView(
            selectedWatchlistId,
            preset.serverId,
            payload
          )
          const nextPreset = toItemsViewPresetFromServer(updated)
          setViewPresets((prev) =>
            normalizeViewPresets(
              prev.map((candidate) =>
                candidate.id === activePresetId ? nextPreset : candidate
              )
            )
          )
          setActivePresetId(nextPreset.id)
          message.success(t("watchlists:items.savedViews.updated", "Saved view updated."))
        } catch (error) {
          console.error("Failed to update item saved view:", error)
          message.error(
            t("watchlists:items.savedViews.updateError", "Failed to update saved view.")
          )
        }
        return
      }

      setViewPresets((prev) =>
        normalizeViewPresets(
          prev.map((candidate) =>
            candidate.id === activePresetId
              ? {
                  ...candidate,
                  sourceId: selectedSourceId,
                  smartFilter,
                  statusFilter,
                  sortMode,
                  searchQuery
                }
              : candidate
          )
        )
      )
      message.success(t("watchlists:items.savedViews.updated", "Saved view updated."))
      return
    }

    setNewViewName(
      t("watchlists:items.savedViews.defaultName", "My view")
    )
    saveViewTriggerRef.current = getFocusableActiveElement()
    setSaveViewModalOpen(true)
  }, [
    activePresetId,
    buildCurrentServerViewPayload,
    normalizeViewPresets,
    searchQuery,
    selectedSourceId,
    selectedWatchlistId,
    smartFilter,
    sortMode,
    statusFilter,
    t,
    viewPresets
  ])

  const openShortcuts = useCallback((source: "button" | "keyboard" = "button") => {
    const activeElement = getFocusableActiveElement()
    const shortcutsButton =
      shortcutsHelpButtonRef.current ||
      (typeof document !== "undefined"
        ? document.querySelector<HTMLButtonElement>(
            "[data-testid='watchlists-items-shortcuts-help']"
          )
        : null)
    if (source === "keyboard") {
      shortcutsTriggerRef.current = shortcutsButton || activeElement
    } else {
      shortcutsTriggerRef.current = activeElement || shortcutsButton
    }
    setShortcutsOpen(true)
  }, [])

  const closeShortcuts = useCallback(() => {
    setShortcutsOpen(false)
  }, [])

  const createViewPreset = useCallback(async () => {
    const trimmedName = newViewName.trim()
    if (!trimmedName) {
      message.error(
        t("watchlists:items.savedViews.nameRequired", "Saved view name is required.")
      )
      return
    }

    if (selectedWatchlistId != null) {
      try {
        const created = await createWatchlistItemView(
          selectedWatchlistId,
          buildCurrentServerViewPayload(trimmedName)
        )
        const createdPreset = toItemsViewPresetFromServer(created)
        setViewPresets((prev) => normalizeViewPresets([createdPreset, ...prev]))
        setActivePresetId(createdPreset.id)
        setSaveViewModalOpen(false)
        setNewViewName("")
        message.success(t("watchlists:items.savedViews.created", "Saved view created."))
      } catch (error) {
        console.error("Failed to create item saved view:", error)
        message.error(
          t("watchlists:items.savedViews.createError", "Failed to create saved view.")
        )
      }
      return
    }

    const newPreset: ItemsViewPreset = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
      name: trimmedName,
      sourceId: selectedSourceId,
      smartFilter,
      statusFilter,
      sortMode,
      searchQuery
    }
    setViewPresets((prev) => normalizeViewPresets([newPreset, ...prev]))
    setActivePresetId(newPreset.id)
    setSaveViewModalOpen(false)
    setNewViewName("")
    message.success(t("watchlists:items.savedViews.created", "Saved view created."))
  }, [
    buildCurrentServerViewPayload,
    newViewName,
    normalizeViewPresets,
    searchQuery,
    selectedSourceId,
    selectedWatchlistId,
    smartFilter,
    sortMode,
    statusFilter,
    t
  ])

  const deleteActiveView = useCallback(() => {
    if (!activePresetId) return
    if (isSystemItemsViewPresetId(activePresetId)) return
    const preset = viewPresets.find((candidate) => candidate.id === activePresetId)
    if (!preset) return
    Modal.confirm({
      title: t("watchlists:items.savedViews.deleteTitle", "Delete saved view?"),
      content: t(
        "watchlists:items.savedViews.deleteDescription",
        "Delete saved view \"{{name}}\"?",
        { name: preset.name }
      ),
      okText: t("common:delete", "Delete"),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", "Cancel"),
      onOk: async () => {
        if (selectedWatchlistId != null && typeof preset.serverId === "number") {
          await deleteWatchlistItemView(selectedWatchlistId, preset.serverId)
        }
        setViewPresets((prev) =>
          normalizeViewPresets(
            prev.filter((candidate) => candidate.id !== activePresetId)
          )
        )
        setActivePresetId((prev) => (prev === activePresetId ? null : prev))
        message.success(t("watchlists:items.savedViews.deleted", "Saved view deleted."))
      }
    })
  }, [activePresetId, normalizeViewPresets, selectedWatchlistId, t, viewPresets])

  const importLegacySavedViews = useCallback(async () => {
    if (selectedWatchlistId == null || legacyViewPresets.length === 0) return
    setImportingLocalViews(true)
    try {
      const results = await Promise.allSettled(
        legacyViewPresets.map((preset) =>
          createWatchlistItemView(
            selectedWatchlistId,
            buildServerItemViewCreatePayload(preset)
          )
        )
      )
      const createdViews = results
        .filter((result): result is PromiseFulfilledResult<Awaited<ReturnType<typeof createWatchlistItemView>>> =>
          result.status === "fulfilled"
        )
        .map((result) => result.value)
      const failedCount = results.length - createdViews.length
      const importedPresets = createdViews.map(toItemsViewPresetFromServer)
      setViewPresets((prev) => normalizeViewPresets([...importedPresets, ...prev]))
      const importedIds = new Set(
        legacyViewPresets
          .filter((_, index) => results[index]?.status === "fulfilled")
          .map((preset) => preset.id)
      )
      setLegacyViewPresets((prev) => prev.filter((preset) => !importedIds.has(preset.id)))
      if (importedPresets.length > 0) {
        message.success(
          t("watchlists:items.savedViews.imported", "Imported {{count}} saved view{{plural}}.", {
            count: importedPresets.length,
            plural: importedPresets.length === 1 ? "" : "s"
          })
        )
      }
      if (failedCount > 0) {
        message.warning(
          t(
            "watchlists:items.savedViews.importPartialError",
            "{{count}} saved view{{plural}} could not be imported and can be retried.",
            {
              count: failedCount,
              plural: failedCount === 1 ? "" : "s"
            }
          )
        )
      }
    } catch (error) {
      console.error("Failed to import local item saved views:", error)
      message.error(
        t("watchlists:items.savedViews.importError", "Failed to import saved views.")
      )
    } finally {
      setImportingLocalViews(false)
    }
  }, [legacyViewPresets, normalizeViewPresets, selectedWatchlistId, t])

  const moveSelectionBy = useCallback((offset: number) => {
    if (sortedItems.length === 0) return
    const currentIndex = sortedItems.findIndex((item) => item.id === selectedItemId)
    const baseIndex =
      currentIndex === -1 ? (offset >= 0 ? 0 : sortedItems.length - 1) : currentIndex
    const nextIndex = Math.max(0, Math.min(sortedItems.length - 1, baseIndex + offset))
    setSelectedItemId(sortedItems[nextIndex]?.id ?? null)
  }, [selectedItemId, sortedItems])

  const openSelectedItemOriginal = useCallback(() => {
    if (!selectedItem?.url) return
    window.open(selectedItem.url, "_blank", "noopener,noreferrer")
  }, [selectedItem])

  const navigateHome = useCallback(() => {
    if (navigate) {
      navigate("/")
      return
    }

    if (typeof window !== "undefined") {
      window.location.hash = "#/"
    }
  }, [navigate])

  const handleChatAboutItem = useCallback(
    (item: ScrapedItem) => {
      const article: WatchlistChatArticle = {
        title: item.title || undefined,
        url: item.url || undefined,
        content: item.content || item.summary || undefined,
        sourceType: "item",
        mediaId: item.media_id ?? undefined
      }
      const payload: WatchlistChatHandoffPayload = { articles: [article] }
      void setSetting(DISCUSS_WATCHLIST_PROMPT_SETTING, payload)
      window.dispatchEvent(
        new CustomEvent("tldw:discuss-watchlist", { detail: payload })
      )
      navigateHome()
    },
    [navigateHome]
  )

  const handleChatAboutSelected = useCallback(() => {
    const selected = items.filter((item) => selectedItemIdSet.has(item.id))
    if (selected.length === 0) return
    const articles: WatchlistChatArticle[] = selected.map((item) => ({
      title: item.title || undefined,
      url: item.url || undefined,
      content: item.content || item.summary || undefined,
      sourceType: "item" as const,
      mediaId: item.media_id ?? undefined
    }))
    const payload: WatchlistChatHandoffPayload = { articles }
    const totalChars = getWatchlistChatTotalChars(payload)
    if (totalChars > WATCHLIST_CHAT_CONTENT_WARN_THRESHOLD) {
      Modal.confirm({
        title: t("watchlists:items.chatSizeWarningTitle", "Large content warning"),
        content: t(
          "watchlists:items.chatSizeWarningContent",
          "Selected updates contain {{chars}} characters of content. This may use significant tokens. Continue with full content?",
          { chars: totalChars.toLocaleString() }
        ),
        okText: t("watchlists:items.chatSizeWarningOk", "Use full content"),
        cancelText: t("watchlists:items.chatSizeWarningCancel", "Cancel"),
        onOk: () => {
          void setSetting(DISCUSS_WATCHLIST_PROMPT_SETTING, payload)
          window.dispatchEvent(
            new CustomEvent("tldw:discuss-watchlist", { detail: payload })
          )
          navigateHome()
        }
      })
      return
    }
    void setSetting(DISCUSS_WATCHLIST_PROMPT_SETTING, payload)
    window.dispatchEvent(
      new CustomEvent("tldw:discuss-watchlist", { detail: payload })
    )
    navigateHome()
  }, [items, navigateHome, selectedItemIdSet, t])

  const openSelectedItemMonitor = useCallback(() => {
    if (!selectedItem) return
    setActiveTab("jobs")
    openJobForm(selectedItem.job_id)
  }, [openJobForm, selectedItem, setActiveTab])

  const openSelectedItemRun = useCallback(() => {
    if (!selectedItem) return
    setRunsJobFilter(selectedItem.job_id)
    setRunsStatusFilter(null)
    setActiveTab("runs")
    openRunDetail(selectedItem.run_id)
  }, [
    openRunDetail,
    selectedItem,
    setActiveTab,
    setRunsJobFilter,
    setRunsStatusFilter
  ])

  const openSelectedItemOutputs = useCallback(() => {
    if (!selectedItem) return
    setOutputsJobFilter(selectedItem.job_id)
    setOutputsRunFilter(selectedItem.run_id)
    setActiveTab("outputs")
  }, [selectedItem, setActiveTab, setOutputsJobFilter, setOutputsRunFilter])

  const openAlertsTab = useCallback(() => {
    setActiveTab("alerts")
  }, [setActiveTab])

  const openQuickCreateFlow = useCallback(() => {
    setActiveTab("sources")
    openSourceForm()
  }, [openSourceForm, setActiveTab])

  const shortcutRows = useMemo(
    () => [
      {
        keys: "j / k",
        description: t(
          "watchlists:items.shortcuts.nextPrevious",
          "Move to next or previous update in the list."
        )
      },
      {
        keys: "space",
        description: t(
          "watchlists:items.shortcuts.toggleReviewed",
          "Toggle reviewed state for the selected update."
        )
      },
      {
        keys: "o",
        description: t(
          "watchlists:items.shortcuts.openOriginal",
          "Open the selected update in a new tab."
        )
      },
      {
        keys: "r",
        description: t(
          "watchlists:items.shortcuts.refresh",
          "Refresh feeds, updates, and smart counts."
        )
      },
      {
        keys: "n",
        description: t(
          "watchlists:items.shortcuts.newFlow",
          "Start a new feed flow by opening Add Source."
        )
      },
      {
        keys: "?",
        description: t(
          "watchlists:items.shortcuts.help",
          "Open this shortcuts help panel."
        )
      }
    ],
    [t]
  )

  const handleShortcutKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.defaultPrevented) return
    if (event.metaKey || event.ctrlKey || event.altKey) return
    if (isEditableTarget(event.target)) return
    if (shortcutsOpen && event.key !== "Escape") return

    if (shortcutsOpen && event.key === "Escape") {
      event.preventDefault()
      closeShortcuts()
      return
    }

    const normalizedKey = event.key.toLowerCase()

    if (normalizedKey === "j") {
      event.preventDefault()
      moveSelectionBy(1)
      return
    }

    if (normalizedKey === "k") {
      event.preventDefault()
      moveSelectionBy(-1)
      return
    }

    if (event.key === " ") {
      if (!selectedItem || updatingItemId !== null || batchReviewScope !== null) return
      event.preventDefault()
      void handleToggleReviewed(selectedItem)
      return
    }

    if (normalizedKey === "o") {
      if (!selectedItem?.url) return
      event.preventDefault()
      openSelectedItemOriginal()
      return
    }

    if (normalizedKey === "r") {
      event.preventDefault()
      refreshItemsView()
      return
    }

    if (normalizedKey === "n") {
      event.preventDefault()
      openQuickCreateFlow()
      return
    }

    const isQuestionMark = event.key === "?" || (event.key === "/" && event.shiftKey)
    if (isQuestionMark) {
      event.preventDefault()
      openShortcuts("keyboard")
    }
  }, [
    batchReviewScope,
    handleToggleReviewed,
    moveSelectionBy,
    openShortcuts,
    openQuickCreateFlow,
    openSelectedItemOriginal,
    refreshItemsView,
    selectedItem,
    closeShortcuts,
    shortcutsOpen,
    updatingItemId
  ])

  useEffect(() => {
    document.addEventListener("keydown", handleShortcutKeyDown)
    return () => document.removeEventListener("keydown", handleShortcutKeyDown)
  }, [handleShortcutKeyDown])

  const smartFeedRows: Array<{ key: SmartFeedFilter; label: string; count: number; icon: React.ReactNode }> = [
    {
      key: "todayUnread",
      label: t("watchlists:items.unreadToday", "Unread today"),
      count: smartCounts.todayUnread,
      icon: <Sun className="h-4 w-4" />
    },
    {
      key: "unread",
      label: t("watchlists:items.unread", "All Unread"),
      count: smartCounts.unread,
      icon: <Rss className="h-4 w-4" />
    },
    {
      key: "today",
      label: t("watchlists:items.today", "Today"),
      count: smartCounts.today,
      icon: <Rss className="h-4 w-4" />
    },
    {
      key: "alertMatches",
      label: t("watchlists:items.alertMatches", "Alert matches"),
      count: smartCounts.alertMatches,
      icon: <Bell className="h-4 w-4" />
    },
    {
      key: "reviewed",
      label: t("watchlists:items.reviewedOnly", "Reviewed"),
      count: smartCounts.reviewed,
      icon: <CheckCircle2 className="h-4 w-4" />
    },
    {
      key: "queued",
      label: t("watchlists:items.queuedForBriefing", "Queued for briefing"),
      count: smartCounts.queued,
      icon: <HelpCircle className="h-4 w-4" />
    },
    {
      key: "all",
      label: t("watchlists:items.filters.all", "All"),
      count: smartCounts.all,
      icon: <RefreshCw className="h-4 w-4" />
    }
  ]

  return (
    <div className="space-y-3" data-testid="watchlists-items-tab">
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        data-testid="watchlists-items-live-region"
      >
        {itemsLiveAnnouncement}
      </div>

      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-text-muted">
          {t(
            "watchlists:items.description",
            "Review collected updates, alert matches, and briefing candidates from this Watchlist."
          )}
        </p>
        <Space>
          {!shortcutsHintVisible && (
            <Button
              size="small"
              onClick={() => setShortcutsHintVisible(true)}
              data-testid="watchlists-items-shortcuts-hint-restore"
            >
              {t("watchlists:items.shortcuts.restoreHint", "Show shortcut hints")}
            </Button>
          )}
          <Tooltip title={t("watchlists:items.shortcuts.helpHint", "Keyboard shortcuts (?)")}>
            <Button
              ref={shortcutsHelpButtonRef}
              icon={<HelpCircle className="h-4 w-4" />}
              onClick={() => openShortcuts("button")}
              data-testid="watchlists-items-shortcuts-help"
            >
              {t("watchlists:items.shortcuts.button", "Shortcuts")}
            </Button>
          </Tooltip>
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={refreshItemsView}
            loading={sourcesLoading || itemsLoading}
          >
            {t("common:refresh", "Refresh")}
          </Button>
        </Space>
      </div>

      {shortcutsHintVisible && (
        <div
          className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface/70 px-3 py-2"
          data-testid="watchlists-items-shortcuts-hint-strip"
        >
          <p className="text-sm text-text-muted">
            {t(
              "watchlists:items.shortcuts.hintStrip",
              "Shortcuts: j/k navigate, space toggles reviewed, o opens original update, ? shows help."
            )}
          </p>
          <Space size="small">
            <Button
              size="small"
              type="link"
              onClick={() => openShortcuts("button")}
              data-testid="watchlists-items-shortcuts-hint-open"
            >
              {t("watchlists:items.shortcuts.openPanel", "View all")}
            </Button>
            <Button
              size="small"
              onClick={() => setShortcutsHintVisible(false)}
              data-testid="watchlists-items-shortcuts-hint-dismiss"
            >
              {t("watchlists:items.shortcuts.dismissHint", "Dismiss")}
            </Button>
          </Space>
        </div>
      )}

      <div
        className="overflow-hidden rounded-2xl border border-border bg-surface"
        data-testid="watchlists-items-layout">
        <div className="grid min-h-[720px] grid-cols-1 xl:grid-cols-[280px_minmax(420px,34vw)_minmax(0,1fr)] 2xl:grid-cols-[300px_minmax(500px,36vw)_minmax(0,1fr)]">
          <aside
            className="border-b border-border bg-surface/70 p-4 xl:border-b-0 xl:border-r"
            aria-label={t("watchlists:items.feedFiltersRegionAria", "Feed filters")}
            data-testid="watchlists-items-left-pane">
            <div className="space-y-4">
              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-subtle">
                  {t("watchlists:items.smartFeeds", "Smart Feeds")}
                </p>
                <div className="space-y-1">
                  {smartFeedRows.map((row) => {
                    const isActive = smartFilter === row.key
                    return (
                      <button
                        key={row.key}
                        type="button"
                      className={`flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-sm transition ${
                          isActive
                            ? "bg-primary/15 text-text"
                            : "text-text-muted hover:bg-surface-hover hover:text-text"
                        }`}
                        onClick={() => handleSmartFilterChange(row.key)}
                        data-testid={`watchlists-items-smart-feed-${row.key}`}
                      >
                        <span className="flex items-center gap-2">
                          {row.icon}
                          <span>{row.label}</span>
                        </span>
                        <span className="text-xs font-semibold tabular-nums text-text-subtle">
                          {row.count}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>

              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-subtle">
                  {t("watchlists:items.feeds", "Feeds")}
                </p>
                <Search
                  placeholder={t("watchlists:items.sourceSearchPlaceholder", "Search feeds...")}
                  value={sourceSearch}
                  onChange={(event) => setSourceSearch(event.target.value)}
                  allowClear
                  className="mb-2"
                />
                <div
                  ref={sourceListRef}
                  className="max-h-[430px] space-y-1 overflow-y-auto pr-1"
                  onScroll={expandVisibleSourcesIfNeeded}
                  role="region"
                  aria-label={t("watchlists:items.feedListAria", "Feeds list")}
                  data-testid="watchlists-items-source-list"
                >
                  <button
                    type="button"
                    data-testid="watchlists-items-source-row-all"
                    className={`flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-sm transition ${
                      selectedSourceId === null
                        ? "bg-primary/15 text-text"
                        : "text-text-muted hover:bg-surface-hover hover:text-text"
                    }`}
                    onClick={() => handleSourceSelect(null)}
                  >
                    <span className="truncate">{t("watchlists:items.allFeeds", "All Feeds")}</span>
                    <span className="text-xs font-semibold tabular-nums text-text-subtle">
                      {smartCounts.all}
                    </span>
                  </button>

                  {sourcesLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Spin size="small" />
                    </div>
                  ) : orderedSources.length === 0 ? (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description={t("watchlists:items.noFeeds", "No feeds found")}
                    />
                  ) : (
                    visibleSources.map((source) => {
                      const selected = selectedSourceId === source.id
                      return (
                        <button
                          key={source.id}
                          type="button"
                          data-testid={`watchlists-items-source-row-${source.id}`}
                          className={`w-full rounded-lg px-2.5 py-2 text-left text-sm transition ${
                            selected
                              ? "bg-primary/15 text-text"
                              : "text-text-muted hover:bg-surface-hover hover:text-text"
                          }`}
                          onClick={() => handleSourceSelect(source.id)}
                        >
                          <div className="truncate font-medium">{source.name}</div>
                          <div className="truncate text-xs text-text-subtle">{source.url}</div>
                        </button>
                      )
                    })
                  )}
                </div>
                {hasCollapsedSources && (
                  <p
                    className="mt-2 text-xs text-text-subtle"
                    data-testid="watchlists-items-source-window-hint"
                  >
                    {t(
                      "watchlists:items.sourceWindowHint",
                      "Showing {{visible}} of {{total}} feeds. Scroll to load more.",
                      {
                        visible: visibleSources.length,
                        total: orderedSources.length
                      }
                    )}
                  </p>
                )}
                {sourcesCappedAtLimit && (
                  <p
                    className="mt-1 text-xs text-amber-700"
                    data-testid="watchlists-items-source-cap-hint"
                  >
                    {t(
                      "watchlists:items.sourceCapHint",
                      "Showing first {{limit}} feeds. Use Feeds tab filters to narrow your source list.",
                      { limit: SOURCE_LOAD_MAX_ITEMS }
                    )}
                  </p>
                )}
              </div>
            </div>
          </aside>

          <section
            className="border-b border-border p-4 xl:border-b-0 xl:border-r"
            aria-label={t("watchlists:items.articleListRegionAria", "Updates list and triage controls")}
            data-testid="watchlists-items-list-pane">
            <div className="mb-3 space-y-2">
              <div className="flex items-end justify-between gap-2">
                <div>
                  <h3 className="text-xl font-semibold text-text">{selectedSourceName}</h3>
                  <p className="text-sm text-text-muted">
                    {t("watchlists:items.unreadCount", "{{count}} unread", {
                      count: currentUnreadCount
                    })}
                  </p>
                </div>
              </div>

              <Search
                placeholder={t("watchlists:items.searchPlaceholder", "Search updates...")}
                value={itemsSearch}
                onChange={(event) => {
                  setStoreItemsSearch(event.target.value)
                  setItemsPage(1)
                }}
                allowClear
              />

              <Segmented<ReaderStatusFilter>
                block
                value={statusFilter}
                onChange={(value) => handleStatusChange(value)}
                options={[
                  { label: t("watchlists:items.filters.all", "All"), value: "all" },
                  {
                    label: t("watchlists:items.filters.ingested", "Ingested"),
                    value: "ingested"
                  },
                  {
                    label: t("watchlists:items.filters.filtered", "Filtered"),
                    value: "filtered"
                  }
                ]}
              />

              {smartFilter === "queued" && (
                <div className="flex flex-wrap items-center gap-2">
                  <Select<number>
                    data-testid="watchlists-items-queue-run-filter"
                    placeholder={t("watchlists:items.queue.runPlaceholder", "Select run")}
                    value={queueRunFilter ?? undefined}
                    onChange={(value) =>
                      handleQueueRunFilterChange(
                        typeof value === "number" ? value : null
                      )
                    }
                    allowClear
                    loading={runsLoading}
                    className="min-w-[200px]"
                    options={queueRunOptions}
                  />
                  <Button
                    size="small"
                    data-testid="watchlists-items-queue-generate-report"
                    loading={queueGenerating}
                    onClick={() => void handleGenerateReportFromQueue()}
                  >
                    {t("watchlists:items.queue.generateReport", "Generate report from queue")}
                  </Button>
                </div>
              )}

              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
                  {t("watchlists:items.sort.label", "Sort")}
                </span>
                <Select<ReaderSortMode>
                  size="small"
                  value={sortMode}
                  options={[
                    {
                      label: t("watchlists:items.sort.newest", "Newest first"),
                      value: "newest"
                    },
                    {
                      label: t("watchlists:items.sort.unreadFirst", "Unread first"),
                      value: "unreadFirst"
                    },
                    {
                      label: t("watchlists:items.sort.oldest", "Oldest first"),
                      value: "oldest"
                    }
                  ]}
                  onChange={(nextSortMode) => handleSortModeChange(nextSortMode)}
                  className="min-w-[170px]"
                  data-testid="watchlists-items-sort-select"
                />
              </div>

              <div className="rounded-lg border border-border bg-surface/70 p-2.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
                    {t("watchlists:items.savedViews.label", "Saved views")}
                  </span>
                  <Select
                    size="small"
                    allowClear
                    value={activePresetId ?? undefined}
                    placeholder={t("watchlists:items.savedViews.placeholder", "Select a saved view")}
                    options={viewPresetOptions}
                    loading={viewPresetsLoading}
                    onChange={(presetId) => {
                      if (!presetId) {
                        setActivePresetId(null)
                        return
                      }
                      applyViewPreset(presetId)
                    }}
                    className="min-w-[220px]"
                    data-testid="watchlists-items-view-presets-select"
                  />
                  <Button
                    size="small"
                    onClick={saveCurrentView}
                    data-testid="watchlists-items-view-save"
                  >
                    {activePresetId && !activePresetIsSystem
                      ? t("watchlists:items.savedViews.update", "Update view")
                      : t("watchlists:items.savedViews.save", "Save view")}
                  </Button>
                  <Button
                    size="small"
                    danger
                    disabled={!activePresetId || activePresetIsSystem}
                    onClick={deleteActiveView}
                    data-testid="watchlists-items-view-delete"
                  >
                    {t("watchlists:items.savedViews.delete", "Delete view")}
                  </Button>
                  {selectedWatchlistId != null && legacyViewPresets.length > 0 && (
                    <Button
                      size="small"
                      onClick={() => void importLegacySavedViews()}
                      loading={importingLocalViews}
                      data-testid="watchlists-items-import-local-views"
                    >
                      {t(
                        "watchlists:items.savedViews.importLocal",
                        "Import local views"
                      )}
                    </Button>
                  )}
                </div>
                <p className="mt-2 text-xs text-text-subtle">
                  {t(
                    "watchlists:items.savedViews.defaultsHint",
                    "Default triage views are pinned. Save changes as a custom view."
                  )}
                </p>
              </div>

              <div className="rounded-lg border border-border bg-surface/70 p-2.5">
                <div className="flex flex-wrap items-center gap-2">
                  <Checkbox
                    checked={allPageItemsSelected}
                    indeterminate={somePageItemsSelected}
                    onChange={(event) => handleToggleSelectPage(event.target.checked)}
                    data-testid="watchlists-items-select-page"
                  >
                    {t("watchlists:items.batch.selectPage", "Select page")}
                  </Checkbox>

                  <span
                    className="text-xs text-text-muted"
                    data-testid="watchlists-items-selected-count"
                  >
                    {t("watchlists:items.batch.selectedCount", "{{count}} selected", {
                      count: selectedItemIds.length
                    })}
                  </span>
                </div>

                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Button
                    size="small"
                    onClick={handleMarkSelectedReviewed}
                    disabled={selectedUnreviewedItemIds.length === 0 || batchReviewScope !== null || collectingAllFiltered}
                    loading={batchReviewScope === "selected"}
                    data-testid="watchlists-items-mark-selected"
                  >
                    {t("watchlists:items.batch.markSelected", "Mark selected as reviewed")}
                  </Button>

                  <Button
                    size="small"
                    onClick={handleMarkPageReviewed}
                    disabled={pageUnreviewedItemIds.length === 0 || batchReviewScope !== null || collectingAllFiltered}
                    loading={batchReviewScope === "page"}
                    data-testid="watchlists-items-mark-page"
                  >
                    {t("watchlists:items.batch.markPage", "Mark page as reviewed")}
                  </Button>

                  <Button
                    size="small"
                    onClick={() => void handleMarkAllFilteredReviewed()}
                    disabled={batchReviewScope !== null}
                    loading={batchReviewScope === "allFiltered" || collectingAllFiltered}
                    data-testid="watchlists-items-mark-all-filtered"
                  >
                    {t(
                      "watchlists:items.batch.markAllFiltered",
                      "Mark all filtered updates"
                    )}
                  </Button>

                  <Button
                    size="small"
                    icon={<MessageSquare className="h-3.5 w-3.5" />}
                    onClick={handleChatAboutSelected}
                    disabled={selectedItemIds.length === 0}
                    data-testid="watchlists-items-chat-selected"
                  >
                    {t("watchlists:items.batch.chatSelected", "Chat about selected ({{count}})", {
                      count: selectedItemIds.length
                    })}
                  </Button>
                </div>

                {collectingAllFiltered && (
                  <p
                    className="mt-2 text-xs text-text-subtle"
                    data-testid="watchlists-items-batch-collecting-all-filtered"
                  >
                    {t(
                      "watchlists:items.batch.collectingAllFiltered",
                      "Collecting unread items that match current filters..."
                    )}
                  </p>
                )}

                {batchReviewProgress && (
                  <div
                    className="mt-2 space-y-2 rounded-lg border border-border bg-surface px-3 py-2"
                    role="status"
                    aria-live="polite"
                    data-testid="watchlists-items-batch-progress"
                  >
                    <span className="sr-only" data-testid="watchlists-items-batch-progress-panel" />
                    <div className="flex items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-text-subtle">
                      <span>
                        {t("watchlists:items.batch.progressLabel", "Batch review progress")}
                      </span>
                      <span data-testid="watchlists-items-batch-progress-count">
                        {batchReviewProgress.processed} / {batchReviewProgress.total}
                      </span>
                    </div>
                    <Progress
                      percent={batchProgressPercent}
                      size="small"
                      status={batchReviewProgress.phase === "failed" ? "exception" : "active"}
                      showInfo={false}
                    />
                    <p
                      className="text-xs text-text-muted"
                      data-testid="watchlists-items-batch-progress-summary"
                    >
                      {batchProgressSummary}
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      {batchReviewScope === null && batchReviewProgress.failedItemIds.length > 0 && (
                        <Button
                          size="small"
                          onClick={retryFailedBatchReview}
                          data-testid="watchlists-items-batch-retry-failed"
                        >
                          {t("watchlists:items.batch.retryFailed", "Retry {{count}} failed", {
                            count: batchReviewProgress.failedItemIds.length
                          })}
                        </Button>
                      )}
                      {batchReviewScope === null && (
                        <Button
                          size="small"
                          type="text"
                          onClick={() => setBatchReviewProgress(null)}
                          data-testid="watchlists-items-batch-progress-dismiss"
                        >
                          {t("watchlists:items.batch.dismissProgress", "Dismiss")}
                        </Button>
                      )}
                    </div>
                  </div>
                )}

                <p
                  className="mt-2 text-xs text-text-subtle"
                  data-testid="watchlists-items-batch-scope-summary"
                >
                  {t(
                    "watchlists:items.batch.scopeSummary",
                    "Selected: {{selected}} unread • This page: {{page}} unread • All filtered: {{allFiltered}} unread.",
                    {
                      selected: selectedUnreviewedCount,
                      page: pageUnreviewedCount,
                      allFiltered: allFilteredUnreadEstimate
                    }
                  )}
                </p>

              </div>
            </div>

            <div
              className="max-h-[560px] space-y-2 overflow-y-auto pr-1"
              role="region"
              aria-label={t("watchlists:items.articleListAria", "Updates list")}
              data-testid="watchlists-items-list">
              {itemsLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Spin />
                </div>
              ) : items.length === 0 ? (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={t("watchlists:items.empty", "No updates found")}
                />
              ) : (
                sortedItems.map((item) => {
                  const selected = item.id === selectedItemId
                  const rowTitle = item.title || t("watchlists:items.untitled", "Untitled item")
                  const sourceLabel =
                    sourceNameById.get(item.source_id) ||
                    t("watchlists:items.unknownSource", "Unknown source")
                  const previewText =
                    itemPreviewById.get(item.id) ||
                    t("watchlists:items.noSummary", "No summary available")
                  const imageUrl = itemImagesById.get(item.id)
                  const reviewStateLabel = item.reviewed
                    ? t("watchlists:items.rowStatusReviewed", "Reviewed")
                    : t("watchlists:items.rowStatusUnread", "Unread")
                  const alertSummary = item.alert_summary
                  const alertSeverity = getHighestSeverity(alertSummary)
                  const rowAriaLabel = t(
                    "watchlists:items.rowAriaLabel",
                    "{{title}} from {{source}}. {{state}}.",
                    {
                      title: rowTitle,
                      source: sourceLabel,
                      state: reviewStateLabel
                    }
                  )

                  return (
                    <button
                      key={item.id}
                      type="button"
                      data-testid={`watchlists-item-row-${item.id}`}
                      aria-label={rowAriaLabel}
                      className={`flex w-full items-start gap-3 rounded-xl border px-3 py-3 text-left transition ${
                        selected
                          ? "border-primary bg-primary/15"
                          : "border-transparent hover:border-border hover:bg-surface-hover"
                      }`}
                      onClick={() => setSelectedItemId(item.id)}
                    >
                      <div className="pt-0.5">
                        <Checkbox
                          checked={selectedItemIdSet.has(item.id)}
                          onChange={(event) =>
                            handleToggleItemSelected(item.id, event.target.checked)
                          }
                          onClick={(event) => event.stopPropagation()}
                          data-testid={`watchlists-item-select-${item.id}`}
                        />
                      </div>

                      <div className="mt-1 shrink-0" aria-hidden="true">
                        {item.reviewed ? (
                          <CheckCircle2
                            className="h-3.5 w-3.5 text-text-subtle"
                            aria-label={t("watchlists:items.rowStatusReviewed", "Reviewed")}
                          />
                        ) : (
                          <span
                            className="block h-2.5 w-2.5 rounded-full bg-primary"
                            role="img"
                            aria-label={t("watchlists:items.rowStatusUnread", "Unread")}
                          />
                        )}
                      </div>

                      {imageUrl ? (
                        <img
                          src={imageUrl}
                          alt={item.title || "article"}
                          className="h-12 w-12 shrink-0 rounded-md object-cover"
                        />
                      ) : (
                        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-md bg-surface-hover text-sm font-semibold uppercase text-text-subtle">
                          {sourceLabel.charAt(0) || "?"}
                        </div>
                      )}

                      <div className="min-w-0 flex-1 space-y-1">
                        <div className="flex items-start justify-between gap-2">
                          <h4 className="line-clamp-2 text-base font-semibold text-text">
                            {rowTitle}
                          </h4>
                          <div className="flex shrink-0 flex-col items-end gap-1">
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${
                                item.reviewed
                                  ? "border-border text-text-subtle"
                                  : "border-primary/40 text-primary"
                              }`}
                              data-testid={`watchlists-item-row-review-state-${item.id}`}
                            >
                              {reviewStateLabel}
                            </span>
                            <span className="text-xs font-medium text-text-subtle">
                              {renderItemTimestamp(item, t)}
                            </span>
                          </div>
                        </div>
                        <p className="line-clamp-2 text-sm text-text-muted">{previewText}</p>
                        {hasAlertSummary(alertSummary) && (
                          <div
                            className="flex flex-wrap items-center gap-1.5 text-xs text-text-subtle"
                            data-testid={`watchlists-item-row-alert-summary-${item.id}`}
                          >
                            <Tag
                              color={alertSeverity ? severityTagColor[alertSeverity] : "default"}
                              className="m-0"
                            >
                              {formatAlertSummaryLabel(alertSummary)}
                            </Tag>
                            {alertSummary?.latest_matched_text && (
                              <span className="line-clamp-1">
                                {alertSummary.latest_matched_text}
                              </span>
                            )}
                          </div>
                        )}
                        <p className="truncate text-xs font-medium text-text-subtle">
                          {sourceLabel}
                        </p>
                      </div>
                    </button>
                  )
                })
              )}
            </div>

            <div className="mt-3">
              <div className="mb-2 flex items-center justify-end gap-2">
                <span className="text-xs text-text-muted">
                  {t("watchlists:items.pageSizeLabel", "Items per page")}
                </span>
                <Select
                  size="small"
                  value={itemsPageSize}
                  options={pageSizeOptions}
                  onChange={(nextPageSize) => {
                    setItemsPage(1)
                    setItemsPageSize(nextPageSize)
                  }}
                  className="min-w-[132px]"
                  data-testid="watchlists-items-page-size-select"
                />
              </div>
              <Pagination
                current={itemsPage}
                pageSize={itemsPageSize}
                total={itemsTotal}
                onChange={(page) => setItemsPage(page)}
                showSizeChanger={false}
                size="small"
                showTotal={(total) =>
                  t("watchlists:items.totalItems", "{{total}} items", { total })
                }
              />
            </div>
          </section>

          <section
            className="min-h-[720px] p-5"
            aria-label={t("watchlists:items.readerRegionAria", "Update reader")}
            data-testid="watchlists-items-reader-pane"
          >
            {itemsLoading && !selectedItem ? (
              <div className="flex h-full items-center justify-center">
                <Spin />
              </div>
            ) : !selectedItem ? (
              <div className="flex h-full items-center justify-center">
                <Empty
                  description={t(
                    "watchlists:items.selectPrompt",
                    "Select an item to view its content"
                  )}
                />
              </div>
            ) : (
              <article className="space-y-5" data-testid="watchlists-item-reader">
                <header className="space-y-3 border-b border-border pb-4">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <p className="text-3xl font-semibold leading-tight text-text">
                        {selectedItem.title ||
                          t("watchlists:items.untitled", "Untitled item")}
                      </p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-text-muted">
                        <Tag>
                          {sourceNameById.get(selectedItem.source_id) ||
                            t("watchlists:items.unknownSource", "Unknown source")}
                        </Tag>
                        <Tag color={selectedItem.status === "ingested" ? "green" : "orange"}>
                          {selectedItem.status}
                        </Tag>
                        {Boolean(selectedItem.queued_for_briefing) && (
                          <Tag color="purple">
                            {t("watchlists:items.queuedTag", "Queued")}
                          </Tag>
                        )}
                        {!selectedItem.reviewed && (
                          <Tag color="blue">{t("watchlists:items.unread", "All Unread")}</Tag>
                        )}
                        {selectedItem.reviewed && (
                          <Tag color="default">
                            {t("watchlists:items.reviewed", "Reviewed")}
                          </Tag>
                        )}
                      </div>
                      <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-text-subtle">
                        {formatReaderDate(selectedItem.published_at || selectedItem.created_at)}
                      </p>
                      {getDomain(selectedItem.url) && (
                        <p className="mt-1 text-sm text-text-muted">
                          {getDomain(selectedItem.url)}
                        </p>
                      )}
                    </div>

                    <Space wrap>
                      <Button
                        size="small"
                        onClick={openSelectedItemMonitor}
                        data-testid="watchlists-item-jump-monitor"
                      >
                        {t("watchlists:items.openMonitor", "Open Monitor")}
                      </Button>
                      <Button
                        size="small"
                        onClick={openSelectedItemRun}
                        data-testid="watchlists-item-jump-run"
                      >
                        {t("watchlists:items.openRunActivity", "Open Run")}
                      </Button>
                      <Button
                        size="small"
                        onClick={openSelectedItemOutputs}
                        data-testid="watchlists-item-jump-outputs"
                      >
                        {t("watchlists:items.openRunOutputs", "View Reports")}
                      </Button>
                      <Button
                        size="small"
                        loading={updatingItemId === selectedItem.id}
                        onClick={() => void handleToggleBriefingQueue(selectedItem)}
                        data-testid="watchlists-item-include-briefing"
                      >
                        {Boolean(selectedItem.queued_for_briefing)
                          ? t(
                              "watchlists:items.removeFromBriefingQueue",
                              "Remove from briefing queue"
                            )
                          : t(
                              "watchlists:items.includeInNextBriefing",
                              "Include in next briefing"
                            )}
                      </Button>
                      {selectedItem.url && (
                        <Tooltip title={selectedItem.url}>
                          <Button
                            size="small"
                            icon={<ExternalLink className="h-3.5 w-3.5" />}
                            onClick={openSelectedItemOriginal}
                          >
                            {t("watchlists:items.openOriginal", "Open Original")}
                          </Button>
                        </Tooltip>
                      )}
                      <Tooltip title={t("watchlists:items.chatAbout", "Chat about this article")}>
                        <Button
                          size="small"
                          icon={<MessageSquare className="h-3.5 w-3.5" />}
                          onClick={() => handleChatAboutItem(selectedItem)}
                          disabled={!selectedItem.content && !selectedItem.summary && !selectedItem.title}
                          data-testid="watchlists-item-chat-about"
                        >
                          {t("watchlists:items.chatAboutButton", "Chat")}
                        </Button>
                      </Tooltip>
                      <Button
                        size="small"
                        loading={updatingItemId === selectedItem.id}
                        onClick={() => void handleToggleReviewed(selectedItem)}
                      >
                        {selectedItem.reviewed
                          ? t(
                              "watchlists:items.markUnreviewed",
                              "Mark as unreviewed"
                            )
                          : t("watchlists:items.markReviewed", "Mark as reviewed")}
                      </Button>
                    </Space>
                  </div>
                </header>

                {hasAlertSummary(selectedItem.alert_summary) && (
                  <div
                    className="rounded-lg border border-amber-200 bg-amber-50/70 px-3 py-3 text-sm text-text"
                    data-testid="watchlists-item-alert-summary-panel"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-amber-700" />
                        <span className="font-semibold">
                          {formatAlertSummaryLabel(selectedItem.alert_summary)}
                        </span>
                        <Tag
                          color={
                            getHighestSeverity(selectedItem.alert_summary)
                              ? severityTagColor[getHighestSeverity(selectedItem.alert_summary)!]
                              : "default"
                          }
                          className="m-0"
                        >
                          {t(
                            "watchlists:items.alertSummary.total",
                            "{{count}} {{label}}",
                            {
                              count: selectedItem.alert_summary?.total || 0,
                              label: selectedItem.alert_summary?.total === 1 ? "match" : "matches"
                            }
                          )}
                        </Tag>
                      </div>
                      <Button
                        size="small"
                        onClick={openAlertsTab}
                        data-testid="watchlists-item-open-alerts"
                      >
                        {t("watchlists:items.alertSummary.openAlerts", "Open Alerts")}
                      </Button>
                    </div>
                    {selectedItem.alert_summary?.latest_matched_text && (
                      <p className="mt-2 text-text-muted">
                        {selectedItem.alert_summary.latest_matched_text}
                      </p>
                    )}
                  </div>
                )}

                {selectedItemPreviewImage && !selectedItemBodyHtml.includes("<img") && (
                  <img
                    src={selectedItemPreviewImage}
                    alt={selectedItem.title || "article image"}
                    className="max-h-[420px] w-full rounded-xl object-cover"
                  />
                )}

                {selectedItemRawBody ? (
                  selectedItemBodyIsHtml ? (
                    <div
                      className="space-y-4 text-[15px] leading-7 text-text [&_a]:text-primary [&_a]:underline [&_h1]:text-3xl [&_h1]:font-semibold [&_h2]:text-2xl [&_h2]:font-semibold [&_img]:my-4 [&_img]:max-h-[460px] [&_img]:w-full [&_img]:rounded-lg [&_img]:object-cover [&_p]:my-3"
                      dangerouslySetInnerHTML={{ __html: selectedItemBodyHtml }}
                    />
                  ) : (
                    <div className="whitespace-pre-wrap text-[15px] leading-7 text-text">
                      {selectedItemRawBody}
                    </div>
                  )
                ) : (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description={t(
                      "watchlists:items.noSummary",
                      "No summary available"
                    )}
                  />
                )}

                {!selectedItem.url && (
                  <Tag>
                    {t(
                      "watchlists:items.noUrlHint",
                      "This item does not include an original URL."
                    )}
                  </Tag>
                )}
              </article>
            )}
          </section>
        </div>
      </div>

      <Modal
        title={t("watchlists:items.savedViews.saveTitle", "Save current view")}
        open={saveViewModalOpen}
        onCancel={() => {
          setSaveViewModalOpen(false)
          setNewViewName("")
        }}
        onOk={createViewPreset}
        okText={t("watchlists:items.savedViews.save", "Save view")}
        cancelText={t("common:cancel", "Cancel")}
        destroyOnHidden
      >
        <div className="space-y-2" data-testid="watchlists-items-view-save-modal">
          <p className="text-sm text-text-muted">
            {t(
              "watchlists:items.savedViews.saveDescription",
              "Save the current source, smart feed, status, and search filters."
            )}
          </p>
          <Input
            value={newViewName}
            onChange={(event) => setNewViewName(event.target.value)}
            placeholder={t("watchlists:items.savedViews.namePlaceholder", "Daily triage")}
            maxLength={64}
            autoFocus
            data-testid="watchlists-items-view-name-input"
          />
        </div>
      </Modal>

      <Modal
        title={t("watchlists:items.shortcuts.title", "Keyboard shortcuts")}
        open={shortcutsOpen}
        onCancel={closeShortcuts}
        footer={null}
        destroyOnHidden
      >
        <div className="space-y-2" data-testid="watchlists-items-shortcuts-modal">
          <div className="flex justify-end">
            <Button
              size="small"
              type="link"
              onClick={closeShortcuts}
              data-testid="watchlists-items-shortcuts-close"
            >
              {t("common:close", "Close")}
            </Button>
          </div>
          {shortcutRows.map((shortcut) => (
            <div
              key={shortcut.keys}
              className="flex items-start justify-between gap-3 rounded-lg border border-border bg-surface/70 px-3 py-2.5"
            >
              <kbd className="rounded-md border border-border bg-surface px-2 py-0.5 font-mono text-xs font-semibold text-text">
                {shortcut.keys}
              </kbd>
              <p className="flex-1 text-sm text-text-muted">{shortcut.description}</p>
            </div>
          ))}
        </div>
      </Modal>
    </div>
  )
}
