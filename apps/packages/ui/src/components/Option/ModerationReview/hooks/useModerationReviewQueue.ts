import * as React from "react"

import {
  decideModerationReviewItem,
  getModerationReviewItem,
  listModerationReviewItems,
  undoModerationReviewDecision,
  type ModerationDecisionAction,
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

export interface ModerationReviewUndoState {
  itemId: string
  token: string
  action: ModerationDecisionAction
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

  const updateFilter = React.useCallback(
    <K extends keyof ModerationReviewFilters>(key: K, value: ModerationReviewFilters[K]) => {
      setFilters((current) => ({
        ...current,
        [key]: value,
        cursor: key === "cursor" ? (value as string | null) : null
      }))
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
        setUndo(response.undo_token ? { itemId: selectedItemId, token: response.undo_token, action } : null)
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
    refresh,
    loadNextPage,
    selectItem,
    decideSelected,
    undoDecision
  }
}
