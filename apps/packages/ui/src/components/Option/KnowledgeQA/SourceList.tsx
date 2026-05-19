/**
 * SourceList - List of retrieved source documents
 */

import React, { useCallback, useMemo, useEffect, useState } from "react"
import { FileText, Keyboard, X } from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { SourceCard, type SourceAskTemplate } from "./SourceCard"
import { cn } from "@/libs/utils"
import type { RagResult } from "./types"
import { getFeedbackSessionId, submitExplicitFeedback } from "@/services/feedback"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { trackKnowledgeQaSearchMetric } from "@/utils/knowledge-qa-search-metrics"
import {
  buildCitationUsageAnchors,
  buildSourceContentFacetCounts,
  buildHighlightTerms,
  buildSourceTypeCounts,
  filterItemsByContentFacet,
  filterItemsByDateRange,
  filterItemsByKeyword,
  filterItemsBySourceType,
  getSourceContentFacetLabel,
  getSourceTypeLabel,
  sortSourceItems,
  type SourceContentFacet,
  type SourceDateFilter,
  type SourceListItem,
  type SourceSortMode,
} from "./sourceListUtils"

const PAGE_SIZE = 10
const SOURCE_LIST_FILTER_STORAGE_PREFIX = "knowledge_qa_source_filters:"
const DEFAULT_SORT_MODE: SourceSortMode = "relevance"
const DEFAULT_SOURCE_TYPE_FILTER = "all"
const DEFAULT_CONTENT_FACET_FILTER: SourceContentFacet = "all"
const DEFAULT_DATE_FILTER: SourceDateFilter = "all"
const SOURCE_SORT_MODES: SourceSortMode[] = ["relevance", "title", "date", "cited"]
const SOURCE_DATE_FILTERS: SourceDateFilter[] = [
  "all",
  "last_30d",
  "last_365d",
  "older_365d",
]
const SOURCE_CONTENT_FACETS: SourceContentFacet[] = [
  "all",
  "pdf",
  "transcript",
  "video",
  "audio",
  "note",
  "web",
  "other",
]

const LazySourceViewerModal = React.lazy(() =>
  import("./SourceViewerModal").then((module) => ({ default: module.SourceViewerModal })),
)

type SourceListProps = {
  className?: string
  layout?: "main" | "rail"
}

type SourceFeedbackState = {
  thumb: "up" | "down" | null
  pendingThumb: "up" | "down" | null
  submitting: boolean
  error: string | null
}

type PersistedSourceListFilters = {
  sortMode: SourceSortMode
  sourceType: string
  contentFacet: SourceContentFacet
  dateFilter: SourceDateFilter
  keyword: string
}

function getSourceFilterStorageKey(threadId: string | null): string {
  const normalizedThreadId =
    typeof threadId === "string" && threadId.trim().length > 0
      ? threadId.trim()
      : "global"
  return `${SOURCE_LIST_FILTER_STORAGE_PREFIX}${normalizedThreadId}`
}

function parsePersistedSourceListFilters(
  rawValue: string
): PersistedSourceListFilters | null {
  try {
    const parsed = JSON.parse(rawValue) as Record<string, unknown>
    if (!parsed || typeof parsed !== "object") {
      return null
    }

    const sortMode = SOURCE_SORT_MODES.includes(parsed.sortMode as SourceSortMode)
      ? (parsed.sortMode as SourceSortMode)
      : DEFAULT_SORT_MODE
    const sourceType =
      typeof parsed.sourceType === "string" && parsed.sourceType.trim().length > 0
        ? parsed.sourceType.trim()
        : DEFAULT_SOURCE_TYPE_FILTER
    const contentFacet = SOURCE_CONTENT_FACETS.includes(
      parsed.contentFacet as SourceContentFacet
    )
      ? (parsed.contentFacet as SourceContentFacet)
      : DEFAULT_CONTENT_FACET_FILTER
    const dateFilter = SOURCE_DATE_FILTERS.includes(parsed.dateFilter as SourceDateFilter)
      ? (parsed.dateFilter as SourceDateFilter)
      : DEFAULT_DATE_FILTER
    const keyword =
      typeof parsed.keyword === "string"
        ? parsed.keyword
        : ""

    return {
      sortMode,
      sourceType,
      contentFacet,
      dateFilter,
      keyword,
    }
  } catch {
    return null
  }
}

function readPersistedSourceListFilters(
  storageKey: string
): PersistedSourceListFilters | null {
  if (typeof window === "undefined") return null

  try {
    const rawStoredValue = window.localStorage.getItem(storageKey)
    return rawStoredValue ? parsePersistedSourceListFilters(rawStoredValue) : null
  } catch (error) {
    console.warn("SourceList filter restore skipped because storage is unavailable.", error)
    return null
  }
}

function persistSourceListFilters(
  storageKey: string,
  payload: PersistedSourceListFilters
): void {
  if (typeof window === "undefined") return

  try {
    window.localStorage.setItem(storageKey, JSON.stringify(payload))
  } catch (error) {
    console.warn("SourceList filter persistence skipped because storage is unavailable.", error)
  }
}

function getResultFeedbackKey(result: RagResult, index: number): string {
  if (typeof result.id === "string" && result.id.length > 0) {
    return result.id
  }
  return `source-${index}`
}

function buildAskPrompt(template: SourceAskTemplate, title: string): string {
  if (template === "summary") {
    return `Summarize ${title}`
  }
  if (template === "quotes") {
    return `Key quotes from ${title}`
  }
  return `Tell me more about ${title}`
}

type PinnedSourceTarget = {
  mediaId: number | null
  noteId: string | null
}

function parseMediaId(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.round(value)
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const normalized = value.trim()
    if (/^\d+$/.test(normalized)) {
      return Number.parseInt(normalized, 10)
    }
  }
  return null
}

function parseNoteId(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(Math.round(value))
  }
  if (typeof value !== "string") return null
  const normalized = value.trim()
  if (!normalized) return null
  const rawId = normalized.startsWith("note_") ? normalized.slice(5).trim() : normalized
  return rawId || null
}

function resolvePinnedSourceTarget(result: RagResult): PinnedSourceTarget {
  const metadata = result.metadata || {}
  const sourceType =
    typeof metadata.source_type === "string"
      ? metadata.source_type.trim().toLowerCase()
      : ""

  if (sourceType === "notes") {
    return {
      mediaId: null,
      noteId:
        parseNoteId(metadata.note_id) ??
        parseNoteId(metadata.id) ??
        parseNoteId(result.id),
    }
  }

  return {
    mediaId:
      parseMediaId(metadata.media_id) ??
      parseMediaId(metadata.id) ??
      parseMediaId(result.id),
    noteId: null,
  }
}

export function SourceList({ className, layout = "main" }: SourceListProps) {
  const {
    results = [],
    citations = [],
    focusedSourceIndex = null,
    focusSource,
    setQuery,
    query = "",
    answer = null,
    searchDetails = null,
    currentThreadId = null,
    messages = [],
    scrollToCitation = () => undefined,
    setPinnedSourceFilters = () => undefined,
  } = useKnowledgeQA()
  const messageApi = useAntdMessage()

  const [sortMode, setSortMode] = React.useState<SourceSortMode>(DEFAULT_SORT_MODE)
  const [activeSourceType, setActiveSourceType] = React.useState<string>(DEFAULT_SOURCE_TYPE_FILTER)
  const [activeContentFacet, setActiveContentFacet] = React.useState<SourceContentFacet>(DEFAULT_CONTENT_FACET_FILTER)
  const [dateFilter, setDateFilter] = React.useState<SourceDateFilter>(DEFAULT_DATE_FILTER)
  const [keywordFilter, setKeywordFilter] = React.useState("")
  const [hydratedFilterStorageKey, setHydratedFilterStorageKey] = React.useState<string | null>(
    null
  )
  const [visibleCount, setVisibleCount] = React.useState(PAGE_SIZE)
  const [shortcutsOpen, setShortcutsOpen] = React.useState(false)
  const [pinnedSources, setPinnedSources] = React.useState<
    Record<string, PinnedSourceTarget>
  >({})
  const [viewerState, setViewerState] = React.useState<{
    result: RagResult | null
    index: number | null
  }>({
    result: null,
    index: null,
  })
  const [feedbackBySource, setFeedbackBySource] = React.useState<
    Record<string, SourceFeedbackState>
  >({})
  const filterStorageKey = useMemo(
    () => getSourceFilterStorageKey(currentThreadId),
    [currentThreadId]
  )
  const activeAnswerSessionKeyRef = React.useRef("")
  const feedbackSessionId = React.useMemo(() => getFeedbackSessionId(), [])
  const latestAssistantMessageId = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "assistant") {
        return messages[index].id
      }
    }
    return null
  }, [messages])
  const answerSessionKey = useMemo(
    () =>
      `${currentThreadId ?? "no-thread"}::${latestAssistantMessageId ?? "no-assistant"}::${
        query.trim() || "no-query"
      }`,
    [currentThreadId, latestAssistantMessageId, query]
  )
  const highlightTerms = useMemo(
    () => buildHighlightTerms(query, searchDetails?.expandedQueries || []),
    [query, searchDetails?.expandedQueries]
  )
  const citationUsageByIndex = useMemo(
    () => buildCitationUsageAnchors(answer),
    [answer]
  )

  const sourceItems = useMemo<SourceListItem[]>(
    () => results.map((result, originalIndex) => ({ result, originalIndex })),
    [results]
  )

  // Get cited indices (0-based)
  const citedIndices = useMemo(
    () => new Set(citations.map((citation) => citation.index - 1)),
    [citations]
  )

  const sourceTypeCounts = useMemo(() => buildSourceTypeCounts(results), [results])
  const contentFacetCounts = useMemo(
    () => buildSourceContentFacetCounts(results),
    [results]
  )

  const sourceTypeFilters = useMemo(() => {
    const options = [
      {
        key: "all",
        label: "All",
        count: results.length,
      },
      ...Object.entries(sourceTypeCounts)
        .filter(([, count]) => count > 0)
        .sort((left, right) => right[1] - left[1])
        .map(([sourceType, count]) => ({
          key: sourceType,
          label: getSourceTypeLabel(sourceType, { plural: true }),
          count,
        })),
    ]

    return options
  }, [results.length, sourceTypeCounts])

  const contentFacetFilters = useMemo(() => {
    const orderedFacets: SourceContentFacet[] = [
      "all",
      "pdf",
      "transcript",
      "video",
      "audio",
      "note",
      "web",
      "other",
    ]

    return orderedFacets
      .map((facet) => ({
        key: facet,
        label: getSourceContentFacetLabel(facet),
        count: facet === "all" ? results.length : contentFacetCounts[facet] || 0,
      }))
      .filter((facet) => facet.key === "all" || facet.count > 0)
  }, [contentFacetCounts, results.length])

  const typeFilteredItems = useMemo(
    () => filterItemsBySourceType(sourceItems, activeSourceType),
    [activeSourceType, sourceItems]
  )
  const contentFilteredItems = useMemo(
    () => filterItemsByContentFacet(typeFilteredItems, activeContentFacet),
    [activeContentFacet, typeFilteredItems]
  )
  const dateFilteredItems = useMemo(
    () => filterItemsByDateRange(contentFilteredItems, dateFilter),
    [contentFilteredItems, dateFilter]
  )
  const filteredItems = useMemo(
    () => filterItemsByKeyword(dateFilteredItems, keywordFilter),
    [dateFilteredItems, keywordFilter]
  )

  const sortedItems = useMemo(
    () => {
      const pinnedSourceKeys = Object.keys(pinnedSources)
      const baseSorted = sortSourceItems(filteredItems, sortMode, citedIndices)
      if (pinnedSourceKeys.length === 0) {
        return baseSorted
      }

      const pinnedSet = new Set(pinnedSourceKeys)
      const pinnedItems = baseSorted.filter((item) =>
        pinnedSet.has(getResultFeedbackKey(item.result, item.originalIndex))
      )
      const unpinnedItems = baseSorted.filter(
        (item) => !pinnedSet.has(getResultFeedbackKey(item.result, item.originalIndex))
      )
      return [...pinnedItems, ...unpinnedItems]
    },
    [citedIndices, filteredItems, pinnedSources, sortMode]
  )

  const visibleItems = useMemo(
    () => sortedItems.slice(0, visibleCount),
    [sortedItems, visibleCount]
  )

  const visibleIndices = useMemo(
    () => visibleItems.map((item) => item.originalIndex),
    [visibleItems]
  )

  const hasMoreResults = visibleItems.length < sortedItems.length
  const activeFilterCount =
    (sortMode !== DEFAULT_SORT_MODE ? 1 : 0) +
    (activeSourceType !== DEFAULT_SOURCE_TYPE_FILTER ? 1 : 0) +
    (activeContentFacet !== DEFAULT_CONTENT_FACET_FILTER ? 1 : 0) +
    (dateFilter !== DEFAULT_DATE_FILTER ? 1 : 0) +
    (keywordFilter.trim().length > 0 ? 1 : 0)
  const hasActiveFilters = activeFilterCount > 0

  const resetFilters = useCallback(() => {
    setSortMode(DEFAULT_SORT_MODE)
    setActiveSourceType(DEFAULT_SOURCE_TYPE_FILTER)
    setActiveContentFacet(DEFAULT_CONTENT_FACET_FILTER)
    setDateFilter(DEFAULT_DATE_FILTER)
    setKeywordFilter("")
  }, [])

  useEffect(() => {
    setVisibleCount(PAGE_SIZE)
  }, [results, activeSourceType, activeContentFacet, dateFilter, keywordFilter, sortMode])

  useEffect(() => {
    const persistedFilters = readPersistedSourceListFilters(filterStorageKey)
    const resolvedFilters: PersistedSourceListFilters = persistedFilters || {
      sortMode: DEFAULT_SORT_MODE,
      sourceType: DEFAULT_SOURCE_TYPE_FILTER,
      contentFacet: DEFAULT_CONTENT_FACET_FILTER,
      dateFilter: DEFAULT_DATE_FILTER,
      keyword: "",
    }

    setSortMode(resolvedFilters.sortMode)
    setActiveSourceType(resolvedFilters.sourceType)
    setActiveContentFacet(resolvedFilters.contentFacet)
    setDateFilter(resolvedFilters.dateFilter)
    setKeywordFilter(resolvedFilters.keyword)
    setHydratedFilterStorageKey(filterStorageKey)
  }, [filterStorageKey])

  useEffect(() => {
    if (hydratedFilterStorageKey !== filterStorageKey) return

    const payload: PersistedSourceListFilters = {
      sortMode,
      sourceType: activeSourceType,
      contentFacet: activeContentFacet,
      dateFilter,
      keyword: keywordFilter,
    }
    persistSourceListFilters(filterStorageKey, payload)
  }, [
    activeContentFacet,
    activeSourceType,
    dateFilter,
    filterStorageKey,
    hydratedFilterStorageKey,
    keywordFilter,
    sortMode,
  ])

  useEffect(() => {
    const validSourceKeys = new Set(
      results.map((result, index) => getResultFeedbackKey(result, index))
    )
    setPinnedSources((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(([sourceKey]) => validSourceKeys.has(sourceKey))
      )
    )
  }, [results])

  useEffect(() => {
    const pinnedTargets = Object.values(pinnedSources)
    const mediaIds = Array.from(
      new Set(
        pinnedTargets
          .map((target) => target.mediaId)
          .filter((value): value is number => value != null)
      )
    )
    const noteIds = Array.from(
      new Set(
        pinnedTargets
          .map((target) => target.noteId)
          .filter((value): value is string => typeof value === "string" && value.length > 0)
      )
    )
    setPinnedSourceFilters({ mediaIds, noteIds })
  }, [pinnedSources, setPinnedSourceFilters])

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target =
        event.target instanceof HTMLElement || event.target instanceof SVGElement
          ? event.target
          : null
      const interactiveCardTarget = Boolean(
        target?.closest(
          'button, a, input, select, textarea, [role="button"], [role="link"], [contenteditable]'
        )
      )
      const isEditableTarget =
        target instanceof HTMLElement &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)

      if (event.key === "?" && !event.metaKey && !event.ctrlKey && !event.altKey) {
        if (isEditableTarget) return
        event.preventDefault()
        setShortcutsOpen(true)
        return
      }

      // Number keys 1-9 to jump to visible sources
      if (
        event.key >= "1" &&
        event.key <= "9" &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey
      ) {
        if (isEditableTarget) return

        const visiblePosition = Number.parseInt(event.key, 10) - 1
        const selectedItem = visibleItems[visiblePosition]
        if (!selectedItem) return

        event.preventDefault()
        focusSource(selectedItem.originalIndex)
        const element = document.getElementById(
          `source-card-${selectedItem.originalIndex}`
        )
        element?.scrollIntoView({ behavior: "smooth", block: "center" })
        return
      }

      // Tab to navigate between visible sources
      if (event.key === "Tab" && !event.shiftKey && visibleIndices.length > 0) {
        const focusedCard = target?.closest('[id^="source-card-"]')
        if (!focusedCard || interactiveCardTarget) return

        event.preventDefault()
        const focusedElementId =
          focusedCard instanceof HTMLElement ? focusedCard.id : null
        const focusedElementIndex = focusedElementId
          ? Number.parseInt(focusedElementId.replace("source-card-", ""), 10)
          : null
        const currentIndex =
          focusedSourceIndex != null ? focusedSourceIndex : focusedElementIndex
        const currentPosition =
          currentIndex != null ? visibleIndices.indexOf(currentIndex) : -1
        const nextPosition =
          currentPosition >= 0
            ? (currentPosition + 1) % visibleIndices.length
            : 0

        const nextIndex = visibleIndices[nextPosition]
        focusSource(nextIndex)
        const element = document.getElementById(`source-card-${nextIndex}`)
        element?.scrollIntoView({ behavior: "smooth", block: "center" })
        return
      }

      // Escape to close shortcut legend or clear focus
      if (event.key === "Escape") {
        if (shortcutsOpen) {
          setShortcutsOpen(false)
          return
        }
        focusSource(null)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [focusSource, focusedSourceIndex, shortcutsOpen, visibleIndices, visibleItems])

  // Handle "Ask About This" - populate query without auto-submitting
  const handleAskAbout = useCallback(
    (result: RagResult, template: SourceAskTemplate) => {
      const title = result.metadata?.title || "this source"
      setQuery(buildAskPrompt(template, title))

      const searchInput =
        (document.getElementById("knowledge-search-input") as HTMLInputElement | null) ||
        (document.querySelector(
          'input[aria-label="Search your knowledge base"]'
        ) as HTMLInputElement | null)
      if (searchInput) {
        searchInput.focus()
        searchInput.setSelectionRange(
          searchInput.value.length,
          searchInput.value.length
        )
      }
    },
    [setQuery]
  )

  const handleViewFull = useCallback((result: RagResult, index: number) => {
    setViewerState({ result, index })
  }, [])

  const closeViewer = useCallback(() => {
    setViewerState({ result: null, index: null })
  }, [])

  const submitSourceFeedback = useCallback(
    async (result: RagResult, resultIndex: number, thumb: "up" | "down") => {
      const requestSessionKey = answerSessionKey
      const sourceKey = getResultFeedbackKey(result, resultIndex)
      const chunkId =
        typeof result.metadata?.chunk_id === "string" &&
        result.metadata.chunk_id.length > 0
          ? result.metadata.chunk_id
          : undefined

      setFeedbackBySource((previous) => ({
        ...previous,
        [sourceKey]: {
          thumb,
          pendingThumb: thumb,
          submitting: true,
          error: null,
        },
      }))

      try {
        await submitExplicitFeedback({
          conversation_id: currentThreadId || undefined,
          message_id: latestAssistantMessageId || undefined,
          query: query.trim() || undefined,
          feedback_type: "relevance",
          relevance_score: thumb === "up" ? 5 : 1,
          document_ids: result.id ? [result.id] : undefined,
          chunk_ids: chunkId ? [chunkId] : undefined,
          session_id: feedbackSessionId,
        })
        if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
          return
        }
        setFeedbackBySource((previous) => ({
          ...previous,
          [sourceKey]: {
            thumb,
            pendingThumb: thumb,
            submitting: false,
            error: null,
          },
        }))
        void trackKnowledgeQaSearchMetric({
          type: "source_feedback_submit",
          relevant: thumb === "up",
        })
      } catch (feedbackError) {
        if (activeAnswerSessionKeyRef.current !== requestSessionKey) {
          return
        }
        const detail =
          feedbackError instanceof Error && feedbackError.message
            ? feedbackError.message
            : "Unable to send source feedback."
        setFeedbackBySource((previous) => ({
          ...previous,
          [sourceKey]: {
            thumb: previous[sourceKey]?.thumb ?? null,
            pendingThumb: thumb,
            submitting: false,
            error: detail,
          },
        }))
        messageApi.open({
          type: "error",
          content: "Source feedback could not be sent. Retry when online.",
          duration: 3,
        })
      }
    },
    [
      currentThreadId,
      feedbackSessionId,
      latestAssistantMessageId,
      messageApi,
      query,
      answerSessionKey,
    ]
  )

  const retrySourceFeedback = useCallback(
    (result: RagResult, resultIndex: number) => {
      const sourceKey = getResultFeedbackKey(result, resultIndex)
      const pendingThumb = feedbackBySource[sourceKey]?.pendingThumb
      if (!pendingThumb) return
      void submitSourceFeedback(result, resultIndex, pendingThumb)
    },
    [feedbackBySource, submitSourceFeedback]
  )

  const togglePinnedSource = useCallback((result: RagResult, sourceIndex: number) => {
    const sourceKey = getResultFeedbackKey(result, sourceIndex - 1)
    setPinnedSources((previous) => {
      if (previous[sourceKey]) {
        const { [sourceKey]: _removed, ...remaining } = previous
        return remaining
      }
      return {
        ...previous,
        [sourceKey]: resolvePinnedSourceTarget(result),
      }
    })
  }, [])

  useEffect(() => {
    activeAnswerSessionKeyRef.current = answerSessionKey
  }, [answerSessionKey])

  useEffect(() => {
    setFeedbackBySource({})
  }, [answerSessionKey, results])

  const [showFiltersOverride, setShowFiltersOverride] = useState(false)
  const isRailLayout = layout === "rail"
  const density: "compact" | "default" | "full" =
    results.length <= 3 ? "compact" : results.length <= 9 ? "default" : "full"
  const showFilters = density === "full" || showFiltersOverride
  const showExtendedFilters = density === "full" || showFiltersOverride

  // Reset filter override when results count changes
  useEffect(() => {
    setShowFiltersOverride(false)
  }, [results.length])

  if (results.length === 0) {
    return null
  }

  return (
    <div className={cn(isRailLayout ? "space-y-3" : "space-y-4", className)}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-text-muted" />
          <h3 className="font-semibold">Sources ({results.length})</h3>
          {Object.keys(pinnedSources).length > 0 ? (
            <span className="inline-flex items-center rounded-md border border-primary/30 bg-primary/10 px-2 py-0.5 text-xs text-primary">
              {Object.keys(pinnedSources).length} pinned
            </span>
          ) : null}
        </div>

        {(density !== "compact" || showFilters) && (
          <div className="flex items-center gap-2">
            {showExtendedFilters && (
              <>
                <label htmlFor="knowledge-source-date-filter" className="sr-only">
                  Filter sources by date range
                </label>
                <select
                  id="knowledge-source-date-filter"
                  value={dateFilter}
                  onChange={(event) => setDateFilter(event.target.value as SourceDateFilter)}
                  className="rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text"
                >
                  <option value="all">Any date</option>
                  <option value="last_30d">Last 30 days</option>
                  <option value="last_365d">Last 12 months</option>
                  <option value="older_365d">Older than 1 year</option>
                </select>
              </>
            )}

            <label htmlFor="knowledge-source-keyword-filter" className="sr-only">
              Filter sources by keyword
            </label>
            <input
              id="knowledge-source-keyword-filter"
              type="search"
              value={keywordFilter}
              onChange={(event) => setKeywordFilter(event.target.value)}
              placeholder="Filter in results"
              className="w-40 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-text placeholder:text-text-muted sm:w-52"
            />

            <label htmlFor="knowledge-source-sort" className="sr-only">
              Sort sources
            </label>
            <select
              id="knowledge-source-sort"
              value={sortMode}
              onChange={(event) => setSortMode(event.target.value as SourceSortMode)}
              className="rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text"
            >
              <option value="relevance">By Relevance</option>
              <option value="title">By Title</option>
              <option value="date">By Date</option>
              <option value="cited">Cited First</option>
            </select>

            <button
              type="button"
              onClick={resetFilters}
              disabled={!hasActiveFilters}
              className={cn(
                "rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
                hasActiveFilters
                  ? "border-border bg-surface text-text-subtle hover:bg-hover hover:text-text"
                  : "border-border bg-surface text-text-muted opacity-60 cursor-not-allowed"
              )}
              aria-label="Reset source filters"
            >
              Reset filters
            </button>
            {hasActiveFilters ? (
              <span className="inline-flex items-center rounded-full border border-primary/30 bg-primary/10 px-2 py-1 text-[11px] font-medium text-primary">
                {activeFilterCount} filter{activeFilterCount === 1 ? "" : "s"} active
              </span>
            ) : null}
          </div>
        )}
      </div>

      {density !== "full" && (
        <button
          type="button"
          onClick={() => setShowFiltersOverride((prev) => !prev)}
          className="text-xs font-medium text-primary hover:text-primaryStrong transition-colors"
        >
          {showFilters ? "Hide filters" : density === "compact" ? "Show filters" : "More filters"}
        </button>
      )}

      {/* Source type filters */}
      {showExtendedFilters && (
        <div className="flex flex-wrap items-center gap-2" aria-label="Source type filters">
          {sourceTypeFilters.map((filter) => (
            <button
              key={filter.key}
              type="button"
              onClick={() => setActiveSourceType(filter.key)}
              aria-pressed={activeSourceType === filter.key}
              className={cn(
                "inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                activeSourceType === filter.key
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border bg-surface text-text-subtle hover:bg-hover hover:text-text"
              )}
            >
              {filter.label}
              <span className="text-[11px] opacity-80">{filter.count}</span>
            </button>
          ))}
        </div>
      )}

      {showExtendedFilters && (
        <div className="flex flex-wrap items-center gap-2" aria-label="Content type filters">
          {contentFacetFilters.map((filter) => (
            <button
              key={filter.key}
              type="button"
              onClick={() => setActiveContentFacet(filter.key)}
              aria-pressed={activeContentFacet === filter.key}
              className={cn(
                "inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                activeContentFacet === filter.key
                  ? "border-primary/40 bg-primary/10 text-primary"
                  : "border-border bg-surface text-text-subtle hover:bg-hover hover:text-text"
              )}
            >
              {filter.label}
              <span className="text-[11px] opacity-80">{filter.count}</span>
            </button>
          ))}
        </div>
      )}

      {/* Keyboard hint + visible count */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-text-muted">
        <span>
          Showing {visibleItems.length} of {sortedItems.length}
          {sortedItems.length !== results.length ? ` (filtered from ${results.length})` : ""} sources
        </span>
        <div className="flex items-center gap-2">
          <span>
            {isRailLayout ? (
              <>
                Jump: <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">1-9</kbd>
                {" • "}
                <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">Tab</kbd> cycles
              </>
            ) : (
              <>
                Press <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">1-9</kbd> to jump,
                <kbd className="ml-1 px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">Tab</kbd> to cycle
              </>
            )}
          </span>
          <button
            type="button"
            onClick={() => setShortcutsOpen(true)}
            className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-text-subtle hover:bg-hover hover:text-text transition-colors"
            aria-label="Open source keyboard shortcuts"
          >
            <Keyboard className="h-3.5 w-3.5" />
            ?
          </button>
        </div>
      </div>

      {visibleItems.length === 0 ? (
        <div className="rounded-lg border border-border bg-muted/20 p-4 text-sm text-text-muted">
          No sources match the selected filters.
        </div>
      ) : (
        <>
          {/* Source cards - consistent 2-column layout above md breakpoint */}
          <div
            data-testid="knowledge-source-grid"
            className={cn(
              "grid gap-4",
              isRailLayout ? "grid-cols-1 gap-3" : "md:grid-cols-2"
            )}
            role="list"
            aria-label="Retrieved sources"
          >
            {visibleItems.map((item) => {
              const sourceKey = getResultFeedbackKey(item.result, item.originalIndex)
              const feedbackState = feedbackBySource[sourceKey] ?? {
                thumb: null,
                pendingThumb: null,
                submitting: false,
                error: null,
              }
              return (
                <SourceCard
                  key={item.result.id || `result-${item.originalIndex}`}
                  result={item.result}
                  index={item.originalIndex + 1}
                  isPinned={Boolean(pinnedSources[sourceKey])}
                  isCited={citedIndices.has(item.originalIndex)}
                  isFocused={focusedSourceIndex === item.originalIndex}
                  onSourceHover={(hoveredIndex) => {
                    focusSource(hoveredIndex)
                  }}
                  highlightTerms={highlightTerms}
                  citationUsages={citationUsageByIndex[item.originalIndex + 1] || []}
                  density={isRailLayout ? "compact" : "default"}
                  onAskAbout={handleAskAbout}
                  onViewFull={handleViewFull}
                  onSourceFeedback={(value, sourceIndex, thumb) => {
                    void submitSourceFeedback(value, sourceIndex - 1, thumb)
                  }}
                  onRetrySourceFeedback={(value, sourceIndex) => {
                    retrySourceFeedback(value, sourceIndex - 1)
                  }}
                  onTogglePin={togglePinnedSource}
                  onJumpToCitation={(citationIndex, occurrence) => {
                    focusSource(item.originalIndex)
                    scrollToCitation(citationIndex, occurrence ?? 1)
                  }}
                  feedbackThumb={feedbackState.thumb}
                  feedbackSubmitting={feedbackState.submitting}
                  feedbackError={feedbackState.error}
                />
              )
            })}
          </div>

          {hasMoreResults && (
            <div className="flex justify-center pt-1">
              <button
                type="button"
                onClick={() => setVisibleCount((previous) => previous + PAGE_SIZE)}
                className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text-subtle hover:bg-hover hover:text-text transition-colors"
              >
                Show more ({sortedItems.length - visibleItems.length} remaining)
              </button>
            </div>
          )}
        </>
      )}

      <React.Suspense fallback={null}>
        <LazySourceViewerModal
          open={Boolean(viewerState.result)}
          result={viewerState.result}
          index={viewerState.index}
          onClose={closeViewer}
        />
      </React.Suspense>

      {shortcutsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <button
            type="button"
            className="absolute inset-0 bg-black/45"
            aria-label="Close source shortcuts"
            onClick={() => setShortcutsOpen(false)}
          />
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Source keyboard shortcuts"
            className="relative w-full max-w-md rounded-xl border border-border bg-surface p-4 shadow-xl"
          >
            <div className="mb-3 flex items-center justify-between">
              <h4 className="text-sm font-semibold">Source keyboard shortcuts</h4>
              <button
                type="button"
                onClick={() => setShortcutsOpen(false)}
                className="rounded p-1 text-text-muted hover:bg-hover hover:text-text"
                aria-label="Close source shortcuts"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <ul className="space-y-2 text-sm text-text-muted">
              <li>
                <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">1-9</kbd> Jump to the
                first nine visible sources.
              </li>
              <li>
                <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">Tab</kbd> Cycle through
                visible source cards.
              </li>
              <li>
                <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">Esc</kbd> Clear source
                focus or close this legend.
              </li>
              <li>
                <kbd className="px-1 py-0.5 bg-bg-subtle text-text rounded font-mono">?</kbd> Open this legend.
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
