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

export function useModerationReviewQueue() {
  const [filters, setFilters] = React.useState<ModerationReviewFilters>(DEFAULT_FILTERS)
  const [items, setItems] = React.useState<ModerationReviewItem[]>([])
  const [selectedItemId, setSelectedItemId] = React.useState<string | null>(null)
  const [selectedItem, setSelectedItem] = React.useState<ModerationReviewItem | null>(null)
  const [total, setTotal] = React.useState<number | null>(null)
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

  const loadFilters = React.useCallback(async (activeFilters: ModerationReviewFilters) => {
    setLoading(true)
    setError(null)
    setPartial(false)
    setWarnings([])
    try {
      const response = await listModerationReviewItems(toListParams(activeFilters))
      const sorted = sortReviewItems(response.items || [], activeFilters.sort)
      setItems(sorted)
      setSelectedIds((current) => new Set([...current].filter((itemId) => sorted.some((item) => item.id === itemId))))
      setTotal(typeof response.total === "number" ? response.total : sorted.length)
      setNextCursor(response.next_cursor || null)
      const preferredId =
        selectedItemId && sorted.some((item) => item.id === selectedItemId)
          ? selectedItemId
          : sorted[0]?.id || null
      setSelectedItemId(preferredId)
      if (preferredId) {
        const fallback = sorted.find((item) => item.id === preferredId) || null
        setSelectedItem(fallback)
        try {
          const detail = await getModerationReviewItem(preferredId)
          setSelectedItem(detail)
        } catch (detailError) {
          setPartial(true)
          setWarnings([`Item detail could not be loaded: ${messageFromError(detailError)}`])
        }
      } else {
        setSelectedItem(null)
      }
    } catch (requestError) {
      setError(requestError)
      setItems([])
      setSelectedItem(null)
      setSelectedItemId(null)
      setTotal(null)
      setNextCursor(null)
    } finally {
      setLoading(false)
    }
  }, [selectedItemId])

  const refresh = React.useCallback(async () => {
    await loadFilters(filters)
  }, [filters, loadFilters])

  const loadNextPage = React.useCallback(async () => {
    if (!nextCursor) {
      return
    }
    const nextFilters = { ...filters, cursor: nextCursor }
    setFilters(nextFilters)
    await loadFilters(nextFilters)
  }, [filters, loadFilters, nextCursor])

  React.useEffect(() => {
    void refresh()
    // Initial load only; subsequent loads are explicit via Refresh to avoid
    // fetching on every search keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selectItem = React.useCallback(
    async (itemId: string) => {
      setSelectedItemId(itemId)
      const fallback = items.find((item) => item.id === itemId) || null
      setSelectedItem(fallback)
      setDetailLoading(true)
      setPartial(false)
      setWarnings([])
      try {
        const detail = await getModerationReviewItem(itemId)
        setSelectedItem(detail)
      } catch (detailError) {
        setPartial(true)
        setWarnings([`Item detail could not be loaded: ${messageFromError(detailError)}`])
      } finally {
        setDetailLoading(false)
      }
    },
    [items]
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
      if (!selectedItemId) {
        return
      }
      setDeciding(action)
      setError(null)
      try {
        const response = await decideModerationReviewItem(selectedItemId, {
          action,
          reason: reason?.trim() || undefined
        })
        setSelectedItem(response.item)
        setUndo(
          response.undo_token
            ? {
                itemId: selectedItemId,
                token: response.undo_token,
                action,
                expiresAt: response.decision.undo_expires_at || null
              }
            : null
        )
        await refresh()
        setSelectedItem(response.item)
        setSelectedItemId(response.item.id)
      } catch (requestError) {
        setError(requestError)
      } finally {
        setDeciding(null)
      }
    },
    [refresh, selectedItemId]
  )

  const undoDecision = React.useCallback(async () => {
    if (!undo) {
      return
    }
    setDeciding("undo")
    setError(null)
    try {
      const item = await undoModerationReviewDecision(undo.itemId, undo.token)
      setSelectedItem(item)
      setSelectedItemId(item.id)
      setUndo(null)
      await refresh()
      setSelectedItem(item)
      setSelectedItemId(item.id)
    } catch (requestError) {
      setError(requestError)
    } finally {
      setDeciding(null)
    }
  }, [refresh, undo])

  const bulkDecideSelected = React.useCallback(
    async (action: ModerationDecisionAction, reason?: string) => {
      if (selectedItemIds.length === 0) {
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
        setBulkResult(response)
        const itemUpdates = new Map(
          response.results
            .filter((entry) => entry.ok && entry.item)
            .map((entry) => [entry.item_id, entry.item as ModerationReviewItem])
        )
        setItems((current) =>
          current
            .map((item) => itemUpdates.get(item.id) || item)
            .filter((item) => filterMatchesActiveStatus(item, filters))
        )
        if (selectedItemId && itemUpdates.has(selectedItemId)) {
          const updated = itemUpdates.get(selectedItemId) || null
          setSelectedItem(updated && filterMatchesActiveStatus(updated, filters) ? updated : null)
        }
        const failedIds = new Set(response.results.filter((entry) => !entry.ok).map((entry) => entry.item_id))
        setSelectedIds(failedIds)
      } catch (requestError) {
        setError(requestError)
      } finally {
        setBulkDeciding(null)
      }
    },
    [filters, selectedItemId, selectedItemIds]
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
