import * as React from "react"

import {
  bulkDecideModerationReviewItems,
  decideModerationReviewItem,
  getModerationReviewItem,
  listModerationReviewItems,
  undoModerationReviewDecision,
  type ModerationDecisionAction,
  type ModerationReviewBulkDecisionResponse,
  type ModerationReviewItem,
  type ModerationReviewListParams,
  type ModerationReviewListResponse,
  type ModerationReviewSort,
  type ModerationReviewStatus,
  type ModerationSeverity
} from "@/services/moderation"
import { sortReviewItems } from "../review-utils"

export interface ModerationReviewFilters {
  status: ModerationReviewStatus | ""
  category: string
  severity: ModerationSeverity | ""
  source_type: string
  source_id: string
  user_id: string
  q: string
  sort: ModerationReviewSort
  cursor: string | null
}

export interface ModerationReviewFilterPreset {
  name: string
  filters: ModerationReviewFilters
}

export interface ModerationReviewUndoState {
  itemId: string
  token: string
  action: ModerationDecisionAction
  expiresAt?: string | null
}

const DEFAULT_FILTERS: ModerationReviewFilters = {
  status: "needs_review",
  category: "",
  severity: "",
  source_type: "",
  source_id: "",
  user_id: "",
  q: "",
  sort: "newest",
  cursor: null
}

const FILTER_PRESETS_STORAGE_KEY = "tldw.moderationReview.filterPresets.v1"

function toListParams(filters: ModerationReviewFilters): ModerationReviewListParams {
  return {
    status: filters.status || undefined,
    category: filters.category.trim() || undefined,
    severity: filters.severity || undefined,
    source_type: filters.source_type.trim() || undefined,
    source_id: filters.source_id.trim() || undefined,
    user_id: filters.user_id.trim() || undefined,
    q: filters.q.trim() || undefined,
    sort: filters.sort,
    cursor: filters.cursor || undefined,
    limit: 50
  }
}

function messageFromError(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return String((error as { message?: unknown })?.message || "Review queue request failed")
}

function loadStoredPresets(): ModerationReviewFilterPreset[] {
  if (typeof window === "undefined") {
    return []
  }
  try {
    const parsed = JSON.parse(window.localStorage.getItem(FILTER_PRESETS_STORAGE_KEY) || "[]")
    if (!Array.isArray(parsed)) {
      return []
    }
    return parsed
      .filter((entry) => entry && typeof entry.name === "string" && entry.filters)
      .map((entry) => ({
        name: entry.name,
        filters: {
          ...DEFAULT_FILTERS,
          ...entry.filters,
          cursor: null
        }
      }))
  } catch {
    return []
  }
}

function persistPresets(presets: ModerationReviewFilterPreset[]) {
  if (typeof window === "undefined") {
    return
  }
  window.localStorage.setItem(FILTER_PRESETS_STORAGE_KEY, JSON.stringify(presets))
}

function filterMatchesActiveStatus(item: ModerationReviewItem, filters: ModerationReviewFilters): boolean {
  return !filters.status || item.status === filters.status
}

type LoadFiltersOptions = {
  append?: boolean
  selectedItemFallback?: ModerationReviewItem | null
}

type ReviewQueuePage = {
  items: ModerationReviewItem[]
  total: number | null
  loadedFilters: ModerationReviewFilters
}

export function useModerationReviewQueue() {
  const mutationVersion = React.useRef(0)
  const queueRead = React.useRef(0)
  const detailRead = React.useRef(0)
  const lifecycle = React.useRef(0)
  const selectionId = React.useRef<string | null>(null)
  const requestedFilters = React.useRef<ModerationReviewFilters>(DEFAULT_FILTERS)
  const [filters, setFilters] = React.useState<ModerationReviewFilters>(DEFAULT_FILTERS)
  const [page, setPageState] = React.useState<ReviewQueuePage>({ items: [], total: null, loadedFilters: DEFAULT_FILTERS })
  // Async acknowledgements need the current page before React commits a render.
  const pageRef = React.useRef(page)
  const setPage = React.useCallback((update: React.SetStateAction<ReviewQueuePage>) => {
    const next = typeof update === "function" ? update(pageRef.current) : update
    pageRef.current = next
    setPageState(next)
  }, [])
  const { items, total, loadedFilters } = page
  const [selectedItemId, setSelectedItemId] = React.useState<string | null>(null)
  const [selectedItem, setSelectedItem] = React.useState<ModerationReviewItem | null>(null)
  const [nextCursor, setNextCursor] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [detailLoading, setDetailLoading] = React.useState(false)
  const [deciding, setDeciding] = React.useState<ModerationDecisionAction | "undo" | null>(null)
  const [error, setError] = React.useState<unknown>(null)
  const [warnings, setWarnings] = React.useState<string[]>([])
  const [partial, setPartial] = React.useState(false)
  const [undo, setUndo] = React.useState<ModerationReviewUndoState | null>(null)
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(() => new Set())
  const [bulkDeciding, setBulkDeciding] = React.useState<ModerationDecisionAction | null>(null)
  const [bulkResult, setBulkResult] = React.useState<ModerationReviewBulkDecisionResponse | null>(null)
  const [filterPresets, setFilterPresets] = React.useState<ModerationReviewFilterPreset[]>(loadStoredPresets)
  const selectedItemIds = React.useMemo(() => Array.from(selectedIds), [selectedIds])
  const selectId = React.useCallback((id: string | null) => {
    selectionId.current = id
    setSelectedItemId(id)
  }, [])
  const recordMutation = React.useCallback((affectedIds: readonly string[]) => {
    mutationVersion.current += 1
    if (selectionId.current && affectedIds.includes(selectionId.current)) {
      detailRead.current += 1
      setDetailLoading(false)
    }
  }, [])

  const updateFilter = React.useCallback(
    <K extends keyof ModerationReviewFilters>(key: K, value: ModerationReviewFilters[K]) => {
      setFilters((current) => ({
        ...current,
        [key]: value,
        cursor: key === "cursor" ? (value as string | null) : null
      }))
      setBulkResult(null)
    },
    []
  )

  const loadFilters = React.useCallback(async (
    activeFilters: ModerationReviewFilters,
    options: LoadFiltersOptions = {}
  ) => {
    requestedFilters.current = activeFilters
    const queueOwner = ++queueRead.current
    const detailOwner = ++detailRead.current
    setDetailLoading(false)
    setLoading(true)
    setError(null)
    setPartial(false)
    setWarnings([])
    try {
      let response: ModerationReviewListResponse
      // Preserve this request intent when an acknowledged write invalidates its GET.
      for (;;) {
        if (queueOwner !== queueRead.current) return
        const version = mutationVersion.current
        try {
          response = await listModerationReviewItems(toListParams(activeFilters))
        } catch (requestError) {
          if (queueOwner !== queueRead.current) return
          if (version !== mutationVersion.current) continue
          throw requestError
        }
        if (queueOwner !== queueRead.current) return
        if (version === mutationVersion.current) break
      }
      const sorted = sortReviewItems(response.items || [], activeFilters.sort)
      const nextItems = options.append
        ? sortReviewItems([
            ...pageRef.current.items.filter((item) => !sorted.some((nextItem) => nextItem.id === item.id)),
            ...sorted
          ], activeFilters.sort)
        : sorted
      setPage({
        items: nextItems,
        total: typeof response.total === "number" ? response.total : nextItems.length,
        loadedFilters: activeFilters
      })
      setSelectedIds((current) => new Set([...current].filter((itemId) => nextItems.some((item) => item.id === itemId))))
      setNextCursor(response.next_cursor || null)
      if (detailOwner !== detailRead.current) return
      const fallbackItem = options.selectedItemFallback || null
      const hasFallbackSelection = Boolean(
        selectionId.current && fallbackItem && fallbackItem.id === selectionId.current
      )
      const hasSelectedItemInPage = Boolean(
        selectionId.current && nextItems.some((item) => item.id === selectionId.current)
      )
      const preferredId = hasSelectedItemInPage || hasFallbackSelection
        ? selectionId.current
        : nextItems[0]?.id || null
      selectId(preferredId)
      if (preferredId) {
        const fallback = nextItems.find((item) => item.id === preferredId) || fallbackItem
        setSelectedItem(fallback)
        try {
          const detail = await getModerationReviewItem(preferredId)
          if (detailOwner !== detailRead.current) return
          setSelectedItem(detail)
        } catch (detailError) {
          if (detailOwner !== detailRead.current) return
          setPartial(true)
          setWarnings([`Item detail could not be loaded: ${messageFromError(detailError)}`])
        }
      } else {
        setSelectedItem(null)
      }
    } catch (requestError) {
      if (queueOwner !== queueRead.current) return
      setError(requestError)
      setPage((current) => ({ ...current, items: [], total: null }))
      if (detailOwner === detailRead.current) {
        setSelectedItem(null)
        selectId(null)
      }
      setNextCursor(null)
    } finally {
      if (queueOwner === queueRead.current) setLoading(false)
    }
  }, [selectId, setPage])

  const refresh = React.useCallback(async (options: LoadFiltersOptions = {}) => {
    await loadFilters(filters, options)
  }, [filters, loadFilters])

  const loadNextPage = React.useCallback(async () => {
    if (!nextCursor) {
      return
    }
    const nextFilters = { ...filters, cursor: nextCursor }
    setFilters(nextFilters)
    await loadFilters(nextFilters, { append: true })
  }, [filters, loadFilters, nextCursor])

  React.useEffect(() => {
    void refresh()
    return () => {
      lifecycle.current += 1
      queueRead.current += 1
      detailRead.current += 1
    }
    // Initial load only; subsequent loads are explicit via Refresh to avoid
    // fetching on every search keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selectItem = React.useCallback(
    async (itemId: string) => {
      const detailOwner = ++detailRead.current
      selectId(itemId)
      const fallback = pageRef.current.items.find((item) => item.id === itemId) || null
      setSelectedItem(fallback)
      setDetailLoading(true)
      setPartial(false)
      setWarnings([])
      try {
        const detail = await getModerationReviewItem(itemId)
        if (detailOwner !== detailRead.current) return
        setSelectedItem(detail)
      } catch (detailError) {
        if (detailOwner !== detailRead.current) return
        setPartial(true)
        setWarnings([`Item detail could not be loaded: ${messageFromError(detailError)}`])
      } finally {
        if (detailOwner === detailRead.current) setDetailLoading(false)
      }
    },
    [selectId]
  )

  const selectRelative = React.useCallback(
    async (direction: 1 | -1) => {
      if (items.length === 0) {
        return
      }
      const currentIndex = selectedItemId
        ? Math.max(0, items.findIndex((item) => item.id === selectedItemId))
        : 0
      const nextIndex = Math.min(items.length - 1, Math.max(0, currentIndex + direction))
      const nextItem = items[nextIndex]
      if (nextItem) {
        await selectItem(nextItem.id)
      }
    },
    [items, selectItem, selectedItemId]
  )

  const toggleSelected = React.useCallback((itemId: string) => {
    setSelectedIds((current) => {
      const next = new Set(current)
      if (next.has(itemId)) {
        next.delete(itemId)
      } else {
        next.add(itemId)
      }
      return next
    })
    setBulkResult(null)
  }, [])

  const clearSelection = React.useCallback(() => {
    setSelectedIds(new Set())
    setBulkResult(null)
  }, [])

  const decideSelected = React.useCallback(
    async (action: ModerationDecisionAction, reason?: string) => {
      const owner = lifecycle.current
      const itemId = selectionId.current
      if (!itemId || error) {
        return
      }
      setDeciding(action)
      setError(null)
      try {
        const response = await decideModerationReviewItem(itemId, {
          action,
          reason: reason?.trim() || undefined
        })
        if (owner !== lifecycle.current) return
        recordMutation([itemId])
        if (selectionId.current === itemId) setSelectedItem(response.item)
        setUndo(
          response.undo_token
            ? {
                itemId,
                token: response.undo_token,
                action,
                expiresAt: response.decision.undo_expires_at || null
              }
            : null
        )
        await loadFilters(requestedFilters.current, { selectedItemFallback: response.item })
      } catch (requestError) {
        if (owner === lifecycle.current) setError(requestError)
      } finally {
        if (owner === lifecycle.current) setDeciding(null)
      }
    },
    [error, loadFilters, recordMutation]
  )

  const undoDecision = React.useCallback(async () => {
    const owner = lifecycle.current
    const target = undo
    if (!target) {
      return
    }
    setDeciding("undo")
    setError(null)
    try {
      const item = await undoModerationReviewDecision(target.itemId, target.token)
      if (owner !== lifecycle.current) return
      recordMutation([target.itemId])
      if (selectionId.current === target.itemId) setSelectedItem(item)
      setUndo((current) => current?.itemId === target.itemId && current.token === target.token ? null : current)
      await loadFilters(requestedFilters.current, { selectedItemFallback: item })
    } catch (requestError) {
      if (owner === lifecycle.current) setError(requestError)
    } finally {
      if (owner === lifecycle.current) setDeciding(null)
    }
  }, [loadFilters, recordMutation, undo])

  const bulkDecideSelected = React.useCallback(
    async (action: ModerationDecisionAction, reason?: string) => {
      const owner = lifecycle.current
      if (selectedItemIds.length === 0 || error) {
        return
      }
      setBulkDeciding(action)
      setError(null)
      try {
        const response = await bulkDecideModerationReviewItems({
          item_ids: selectedItemIds,
          action,
          reason: reason?.trim() || undefined
        })
        if (owner !== lifecycle.current) return
        const succeededIds = response.results.filter((entry) => entry.ok).map((entry) => entry.item_id)
        if (succeededIds.length) recordMutation(succeededIds)
        setBulkResult(response)
        const itemUpdates = new Map(
          response.results
            .filter((entry) => entry.ok && entry.item)
            .map((entry) => [entry.item_id, entry.item as ModerationReviewItem])
        )
        setPage((current) => {
          const removedIds = new Set(current.items.filter((item) => {
            const updated = itemUpdates.get(item.id)
            return updated && !filterMatchesActiveStatus(updated, current.loadedFilters)
          }).map((item) => item.id))
          return {
            ...current,
            items: current.items
              .map((item) => itemUpdates.get(item.id) || item)
              .filter((item) => !removedIds.has(item.id)),
            total: current.total === null ? null : Math.max(0, current.total - removedIds.size)
          }
        })
        const currentItemId = selectionId.current
        if (currentItemId && itemUpdates.has(currentItemId)) {
          const updated = itemUpdates.get(currentItemId) || null
          setSelectedItem(updated && filterMatchesActiveStatus(updated, pageRef.current.loadedFilters) ? updated : null)
        }
        setSelectedIds((current) => new Set([...current].filter((id) => !succeededIds.includes(id))))
        if (response.results.some((entry) => entry.ok && !entry.item)) {
          if (selectionId.current && succeededIds.includes(selectionId.current)) {
            selectId(null)
            setSelectedItem(null)
          }
          throw new Error("Some decisions were accepted without updated items. Refresh the queue before reviewing again.")
        }
      } catch (requestError) {
        if (owner === lifecycle.current) setError(requestError)
      } finally {
        if (owner === lifecycle.current) setBulkDeciding(null)
      }
    },
    [error, recordMutation, selectedItemIds, selectId, setPage]
  )

  const saveFilterPreset = React.useCallback(
    (name: string) => {
      const normalized = name.trim()
      if (!normalized) {
        return
      }
      const { cursor: _cursor, ...presetFilters } = filters
      const next = [
        ...filterPresets.filter((preset) => preset.name !== normalized),
        { name: normalized, filters: { ...presetFilters, cursor: null } }
      ].sort((a, b) => a.name.localeCompare(b.name))
      setFilterPresets(next)
      persistPresets(next)
    },
    [filterPresets, filters]
  )

  const applyFilterPreset = React.useCallback(
    (name: string) => {
      const preset = filterPresets.find((entry) => entry.name === name)
      if (!preset) {
        return
      }
      setFilters({
        ...DEFAULT_FILTERS,
        ...preset.filters,
        cursor: null
      })
      setBulkResult(null)
    },
    [filterPresets]
  )

  const deleteFilterPreset = React.useCallback(
    (name: string) => {
      const next = filterPresets.filter((entry) => entry.name !== name)
      setFilterPresets(next)
      persistPresets(next)
    },
    [filterPresets]
  )

  return {
    filters,
    loadedFilters,
    updateFilter,
    items,
    selectedItemId,
    selectedItem,
    total,
    nextCursor,
    loading,
    detailLoading,
    deciding,
    error,
    warnings,
    partial,
    undo,
    selectedItemIds,
    bulkDeciding,
    bulkResult,
    filterPresets,
    refresh,
    loadNextPage,
    selectItem,
    selectRelative,
    toggleSelected,
    clearSelection,
    decideSelected,
    undoDecision,
    bulkDecideSelected,
    saveFilterPreset,
    applyFilterPreset,
    deleteFilterPreset
  }
}
