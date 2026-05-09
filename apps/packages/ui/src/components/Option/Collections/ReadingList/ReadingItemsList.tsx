import React, { Suspense, useCallback, useEffect, useMemo, useState } from "react"
import {
  Button,
  Checkbox,
  Dropdown,
  Empty,
  Input,
  Modal,
  Pagination,
  Select,
  Spin,
  message
} from "antd"
import type { MenuProps } from "antd"
import { Filter, Plus, RefreshCw, Search, Star, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { useCollectionsStore } from "@/store/collections"
import type { ReadingSavedSearch, ReadingStatus } from "@/types/collections"
import { formatDateInputValue, parseDateInputValue } from "@/utils/date-input"
import { normalizeBulkTags, getBulkFailureLines } from "./bulkActions"
import { ReadingItemCard } from "./ReadingItemCard"
import { ReadingItemDetail } from "./ReadingItemDetail"
import { SavedSearchesMenu } from "./SavedSearchesMenu"

const STATUS_OPTIONS: { value: ReadingStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "saved", label: "Saved" },
  { value: "reading", label: "Reading" },
  { value: "read", label: "Read" },
  { value: "archived", label: "Archived" }
]

const SORT_OPTIONS = [
  { value: "created_at", label: "Date Added" },
  { value: "updated_at", label: "Last Updated" },
  { value: "title", label: "Title" },
  { value: "relevance", label: "Relevance" }
]

const AddUrlModal = React.lazy(() =>
  import("./AddUrlModal").then((m) => ({ default: m.AddUrlModal }))
)

const normalizeReadingStatus = (value: unknown): ReadingStatus => {
  if (value === "saved" || value === "reading" || value === "read" || value === "archived") {
    return value
  }
  return "saved"
}

export const ReadingItemsList: React.FC = () => {
  const { t } = useTranslation(["collections", "common"])
  const api = useTldwApiClient()
  const { showUndoNotification } = useUndoNotification()

  // Store state
  const items = useCollectionsStore((s) => s.items)
  const itemsLoading = useCollectionsStore((s) => s.itemsLoading)
  const itemsError = useCollectionsStore((s) => s.itemsError)
  const itemsTotal = useCollectionsStore((s) => s.itemsTotal)
  const itemsPage = useCollectionsStore((s) => s.itemsPage)
  const itemsPageSize = useCollectionsStore((s) => s.itemsPageSize)
  const itemsSearch = useCollectionsStore((s) => s.itemsSearch)
  const filterStatus = useCollectionsStore((s) => s.filterStatus)
  const filterFavorite = useCollectionsStore((s) => s.filterFavorite)
  const filterTags = useCollectionsStore((s) => s.filterTags)
  const filterDomain = useCollectionsStore((s) => s.filterDomain)
  const filterDateFrom = useCollectionsStore((s) => s.filterDateFrom)
  const filterDateTo = useCollectionsStore((s) => s.filterDateTo)
  const sortBy = useCollectionsStore((s) => s.sortBy)
  const sortOrder = useCollectionsStore((s) => s.sortOrder)
  const availableTags = useCollectionsStore((s) => s.availableTags)
  const itemDetailOpen = useCollectionsStore((s) => s.itemDetailOpen)
  const addUrlModalOpen = useCollectionsStore((s) => s.addUrlModalOpen)
  const readingSavedSearchesEnabled = useCollectionsStore((s) => s.readingSavedSearchesEnabled)
  const deleteConfirmOpen = useCollectionsStore((s) => s.deleteConfirmOpen)
  const deleteTargetId = useCollectionsStore((s) => s.deleteTargetId)
  const deleteTargetType = useCollectionsStore((s) => s.deleteTargetType)

  // Store actions
  const setItems = useCollectionsStore((s) => s.setItems)
  const setItemsLoading = useCollectionsStore((s) => s.setItemsLoading)
  const setItemsError = useCollectionsStore((s) => s.setItemsError)
  const setItemsPage = useCollectionsStore((s) => s.setItemsPage)
  const setItemsSearch = useCollectionsStore((s) => s.setItemsSearch)
  const setFilterStatus = useCollectionsStore((s) => s.setFilterStatus)
  const setFilterFavorite = useCollectionsStore((s) => s.setFilterFavorite)
  const setFilterTags = useCollectionsStore((s) => s.setFilterTags)
  const setFilterDomain = useCollectionsStore((s) => s.setFilterDomain)
  const setFilterDateRange = useCollectionsStore((s) => s.setFilterDateRange)
  const setSortBy = useCollectionsStore((s) => s.setSortBy)
  const setSortOrder = useCollectionsStore((s) => s.setSortOrder)
  const openAddUrlModal = useCollectionsStore((s) => s.openAddUrlModal)
  const resetFilters = useCollectionsStore((s) => s.resetFilters)
  const setReadingSavedSearchesEnabled = useCollectionsStore((s) => s.setReadingSavedSearchesEnabled)
  const closeDeleteConfirm = useCollectionsStore((s) => s.closeDeleteConfirm)
  const removeItem = useCollectionsStore((s) => s.removeItem)
  const setAvailableTags = useCollectionsStore((s) => s.setAvailableTags)

  const [deleteLoading, setDeleteLoading] = useState(false)
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([])
  const [bulkActionLoading, setBulkActionLoading] = useState(false)
  const [tagModalOpen, setTagModalOpen] = useState(false)
  const [tagActionMode, setTagActionMode] = useState<"add" | "remove">("add")
  const [tagInput, setTagInput] = useState("")
  const [outputModalOpen, setOutputModalOpen] = useState(false)
  const [outputTemplateId, setOutputTemplateId] = useState<string | null>(null)
  const [outputTitle, setOutputTitle] = useState("")
  const [outputTemplates, setOutputTemplates] = useState<Array<{ label: string; value: string }>>([])
  const [outputTemplatesLoading, setOutputTemplatesLoading] = useState(false)
  const [outputGenerating, setOutputGenerating] = useState(false)
  const [savedSearchesLoading, setSavedSearchesLoading] = useState(false)
  const [savedSearches, setSavedSearches] = useState<ReadingSavedSearch[]>([])

  const selectedCount = selectedItemIds.length
  const allPageSelected = items.length > 0 && selectedCount === items.length

  const buildSortParam = useCallback(() => {
    if (sortBy === "relevance") return "relevance"
    const direction = sortOrder === "asc" ? "asc" : "desc"
    if (sortBy === "created_at") return `created_${direction}`
    if (sortBy === "updated_at") return `updated_${direction}`
    if (sortBy === "title") return `title_${direction}`
    return undefined
  }, [sortBy, sortOrder])

  const applySavedSearch = useCallback(
    (search: { name: string; query: Record<string, unknown>; sort?: string }) => {
      const query = search.query || {}
      const statusRaw = query.status
      let nextStatus: ReadingStatus | "all" = "all"
      if (typeof statusRaw === "string") {
        nextStatus = statusRaw as ReadingStatus
      } else if (Array.isArray(statusRaw) && typeof statusRaw[0] === "string") {
        nextStatus = statusRaw[0] as ReadingStatus
      }
      if (!["saved", "reading", "read", "archived", "all"].includes(nextStatus)) {
        nextStatus = "all"
      }

      const tagsRaw = query.tags
      const nextTags = Array.isArray(tagsRaw)
        ? tagsRaw.map((tag) => String(tag).trim()).filter(Boolean)
        : []
      const favoriteRaw = query.favorite
      const domainRaw = query.domain
      const qRaw = query.q
      const dateFromRaw = query.date_from
      const dateToRaw = query.date_to
      const resolvedSort = search.sort || (typeof query.sort === "string" ? query.sort : undefined)

      setItemsSearch(typeof qRaw === "string" ? qRaw : "")
      setFilterStatus(nextStatus)
      setFilterFavorite(typeof favoriteRaw === "boolean" ? favoriteRaw : null)
      setFilterTags(nextTags)
      setFilterDomain(typeof domainRaw === "string" ? domainRaw : "")
      setFilterDateRange(
        typeof dateFromRaw === "string" ? dateFromRaw : null,
        typeof dateToRaw === "string" ? dateToRaw : null
      )
      if (resolvedSort === "updated_desc") {
        setSortBy("updated_at")
        setSortOrder("desc")
      } else if (resolvedSort === "updated_asc") {
        setSortBy("updated_at")
        setSortOrder("asc")
      } else if (resolvedSort === "created_desc") {
        setSortBy("created_at")
        setSortOrder("desc")
      } else if (resolvedSort === "created_asc") {
        setSortBy("created_at")
        setSortOrder("asc")
      } else if (resolvedSort === "title_desc") {
        setSortBy("title")
        setSortOrder("desc")
      } else if (resolvedSort === "title_asc") {
        setSortBy("title")
        setSortOrder("asc")
      } else if (resolvedSort === "relevance") {
        setSortBy("relevance")
        setSortOrder("desc")
      }
      message.success(
        t("collections:reading.savedSearchApplied", "Applied saved search: {{name}}", {
          name: search.name
        })
      )
    },
    [
      setFilterDateRange,
      setFilterDomain,
      setFilterFavorite,
      setFilterStatus,
      setFilterTags,
      setItemsSearch,
      setSortBy,
      setSortOrder,
      t
    ]
  )

  const fetchSavedSearches = useCallback(async () => {
    if (!readingSavedSearchesEnabled) return
    setSavedSearchesLoading(true)
    try {
      const response = await api.listReadingSavedSearches({ limit: 100, offset: 0 })
      setSavedSearches(response.items || [])
    } catch (error: any) {
      const errorMsg = error?.message || "Failed to load saved searches"
      if (errorMsg.includes("reading_saved_searches_disabled")) {
        setReadingSavedSearchesEnabled(false)
        setSavedSearches([])
      } else {
        message.error(errorMsg)
      }
    } finally {
      setSavedSearchesLoading(false)
    }
  }, [api, readingSavedSearchesEnabled, setReadingSavedSearchesEnabled])

  const createSavedSearchFromCurrent = useCallback(async () => {
    if (!readingSavedSearchesEnabled) return
    const query: Record<string, unknown> = {}
    if (itemsSearch.trim()) query.q = itemsSearch.trim()
    if (filterStatus !== "all") query.status = [filterStatus]
    if (typeof filterFavorite === "boolean") query.favorite = filterFavorite
    if (filterTags.length > 0) query.tags = filterTags
    if (filterDomain.trim()) query.domain = filterDomain.trim()
    if (filterDateFrom) query.date_from = filterDateFrom
    if (filterDateTo) query.date_to = filterDateTo
    const sort = buildSortParam()
    if (sort) query.sort = sort

    const defaultName = itemsSearch.trim()
      ? t("collections:reading.savedSearchNamed", "Search: {{query}}", { query: itemsSearch.trim() })
      : t("collections:reading.savedSearchDefaultName", "Saved Search")

    try {
      const created = await api.createReadingSavedSearch({
        name: defaultName,
        query,
        sort
      })
      setSavedSearches((prev) => {
        const deduped = prev.filter((entry) => entry.id !== created.id)
        return [created, ...deduped]
      })
      message.success(
        t("collections:reading.savedSearchCreated", "Saved current filters")
      )
    } catch (error: any) {
      const errorMsg = error?.message || "Failed to save search"
      if (errorMsg.includes("reading_saved_searches_disabled")) {
        setReadingSavedSearchesEnabled(false)
        setSavedSearches([])
      } else {
        message.error(errorMsg)
      }
    }
  }, [
    api,
    buildSortParam,
    filterDateFrom,
    filterDateTo,
    filterDomain,
    filterFavorite,
    filterStatus,
    filterTags,
    itemsSearch,
    readingSavedSearchesEnabled,
    setReadingSavedSearchesEnabled,
    t
  ])

  // Fetch items
  const fetchItems = useCallback(async () => {
    setItemsLoading(true)
    setItemsError(null)
    try {
      const response = await api.getReadingList({
        page: itemsPage,
        size: itemsPageSize,
        q: itemsSearch || undefined,
        status: filterStatus !== "all" ? filterStatus : undefined,
        favorite: filterFavorite ?? undefined,
        tags: filterTags.length ? filterTags : undefined,
        domain: filterDomain || undefined,
        date_from: filterDateFrom || undefined,
        date_to: filterDateTo || undefined,
        sort: buildSortParam()
      })
      setItems(response.items, response.total)
      const tagSet = new Set<string>()
      response.items.forEach((item: { tags?: string[] }) => {
        item.tags?.forEach((tag) => tagSet.add(tag))
      })
      setAvailableTags(Array.from(tagSet).sort())
    } catch (error: any) {
      const errorMsg = error?.message || "Failed to fetch reading list"
      setItemsError(errorMsg)
      message.error(errorMsg)
    } finally {
      setItemsLoading(false)
    }
  }, [
    api,
    itemsPage,
    itemsPageSize,
    itemsSearch,
    filterStatus,
    filterFavorite,
    filterTags,
    filterDomain,
    filterDateFrom,
    filterDateTo,
    buildSortParam,
    setItems,
    setItemsLoading,
    setItemsError,
    setAvailableTags
  ])

  // Fetch on mount and when filters change
  useEffect(() => {
    fetchItems()
  }, [fetchItems])

  useEffect(() => {
    void fetchSavedSearches()
  }, [fetchSavedSearches])

  // Keep selection limited to the current page list.
  useEffect(() => {
    const pageIds = new Set(items.map((item) => item.id))
    setSelectedItemIds((prev) => prev.filter((id) => pageIds.has(id)))
  }, [items])

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setItemsSearch(e.target.value)
    },
    [setItemsSearch]
  )

  const handlePageChange = useCallback(
    (page: number) => {
      setItemsPage(page)
    },
    [setItemsPage]
  )

  const handleTagFilterChange = useCallback(
    (tags: string[]) => {
      const normalized = tags
        .map((tag) => tag.trim().toLowerCase())
        .filter(Boolean)
      setFilterTags(Array.from(new Set(normalized)))
    },
    [setFilterTags]
  )

  const handleDomainChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setFilterDomain(e.target.value)
    },
    [setFilterDomain]
  )

  const handleDateFromChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setFilterDateRange(
        parseDateInputValue(event.target.value, "start"),
        filterDateTo
      )
    },
    [filterDateTo, setFilterDateRange]
  )

  const handleDateToChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setFilterDateRange(
        filterDateFrom,
        parseDateInputValue(event.target.value, "end")
      )
    },
    [filterDateFrom, setFilterDateRange]
  )

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTargetId || deleteTargetType !== "item") return
    setDeleteLoading(true)
    try {
      await api.deleteReadingItem(deleteTargetId, { hard: true })
      removeItem(deleteTargetId)
      message.success(t("collections:reading.deleted", "Article deleted"))
    } catch (error: any) {
      message.error(error?.message || "Failed to delete article")
    } finally {
      setDeleteLoading(false)
      closeDeleteConfirm()
    }
  }, [api, deleteTargetId, deleteTargetType, removeItem, closeDeleteConfirm, t])

  const applyBulkAction = useCallback(
    async (
      payload: {
        action: "set_status" | "set_favorite" | "add_tags" | "remove_tags" | "replace_tags" | "delete"
        status?: ReadingStatus
        favorite?: boolean
        tags?: string[]
        hard?: boolean
      },
      actionLabel: string
    ) => {
      if (selectedItemIds.length === 0) {
        message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
        return false
      }

      setBulkActionLoading(true)
      try {
        const response = await api.bulkUpdateReadingItems({
          item_ids: selectedItemIds,
          ...payload
        })

        const succeededIds = new Set(
          response.results
            .filter((entry) => entry.success)
            .map((entry) => entry.item_id)
        )
        if (succeededIds.size > 0) {
          setSelectedItemIds((prev) => prev.filter((id) => !succeededIds.has(id)))
          await fetchItems()
        }

        if (response.failed > 0) {
          const failureLines = getBulkFailureLines(response)
          Modal.info({
            title: t("collections:reading.bulk.summaryTitle", "Bulk action summary"),
            width: 620,
            content: (
              <div className="space-y-2">
                <p>
                  {t(
                    "collections:reading.bulk.summaryBody",
                    "{{action}} completed. {{succeeded}} succeeded, {{failed}} failed.",
                    {
                      action: actionLabel,
                      succeeded: response.succeeded,
                      failed: response.failed
                    }
                  )}
                </p>
                {failureLines.length > 0 && (
                  <div className="max-h-52 overflow-auto rounded border border-border bg-bg p-2 text-xs">
                    {failureLines.map((line) => (
                      <div key={line}>{line}</div>
                    ))}
                    {response.failed > failureLines.length && (
                      <div>
                        {t("collections:reading.bulk.moreFailures", "+{{count}} more failures", {
                          count: response.failed - failureLines.length
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })
        } else {
          message.success(
            t("collections:reading.bulk.success", "{{action}} applied to {{count}} items", {
              action: actionLabel,
              count: response.succeeded
            })
          )
        }
        return true
      } catch (error: any) {
        message.error(error?.message || "Bulk action failed")
        return false
      } finally {
        setBulkActionLoading(false)
      }
    },
    [api, fetchItems, selectedItemIds, t]
  )

  const handleItemSelectionChange = useCallback((itemId: string, checked: boolean) => {
    setSelectionMode(true)
    setSelectedItemIds((prev) => {
      if (checked) {
        if (prev.includes(itemId)) return prev
        return [...prev, itemId]
      }
      return prev.filter((id) => id !== itemId)
    })
  }, [])

  const handleSelectionModeToggle = useCallback(() => {
    if (selectionMode) {
      setSelectionMode(false)
      setSelectedItemIds([])
      return
    }
    setSelectionMode(true)
  }, [selectionMode])

  const handleSelectAllToggle = useCallback(() => {
    if (allPageSelected) {
      setSelectedItemIds([])
      return
    }
    setSelectionMode(true)
    setSelectedItemIds(items.map((item) => item.id))
  }, [allPageSelected, items])

  const handleBulkStatus = useCallback(
    async (status: ReadingStatus) => {
      const statusLabel = t(`collections:status.${status}`, status)
      await applyBulkAction(
        { action: "set_status", status },
        t("collections:reading.bulk.actions.setStatusLabel", "Set status: {{status}}", {
          status: statusLabel
        })
      )
    },
    [applyBulkAction, t]
  )

  const handleBulkFavorite = useCallback(
    async (favorite: boolean) => {
      await applyBulkAction(
        { action: "set_favorite", favorite },
        favorite
          ? t("collections:reading.bulk.actions.favoriteOn", "Mark favorite")
          : t("collections:reading.bulk.actions.favoriteOff", "Unfavorite")
      )
    },
    [applyBulkAction, t]
  )

  const openTagActionModal = useCallback(
    (mode: "add" | "remove") => {
      if (selectedItemIds.length === 0) {
        message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
        return
      }
      setTagActionMode(mode)
      setTagInput("")
      setTagModalOpen(true)
    },
    [selectedItemIds.length, t]
  )

  const handleTagActionConfirm = useCallback(async () => {
    const tags = normalizeBulkTags(tagInput)
    if (tags.length === 0) {
      message.warning(t("collections:reading.bulk.tagsRequired", "Enter at least one tag"))
      return
    }
    const action = tagActionMode === "add" ? "add_tags" : "remove_tags"
    const label =
      tagActionMode === "add"
        ? t("collections:reading.bulk.actions.addTags", "Add tags")
        : t("collections:reading.bulk.actions.removeTags", "Remove tags")
    const ok = await applyBulkAction({ action, tags }, label)
    if (ok) {
      setTagModalOpen(false)
      setTagInput("")
    }
  }, [applyBulkAction, tagActionMode, tagInput, t])

  const restoreDeletedItems = useCallback(
    async (statusByItemId: Record<string, ReadingStatus>) => {
      const grouped = new Map<ReadingStatus, string[]>()
      Object.entries(statusByItemId).forEach(([itemId, status]) => {
        const normalizedStatus = normalizeReadingStatus(status)
        const existing = grouped.get(normalizedStatus)
        if (existing) {
          existing.push(itemId)
        } else {
          grouped.set(normalizedStatus, [itemId])
        }
      })

      let failed = 0
      for (const [status, itemIds] of grouped.entries()) {
        const response = await api.bulkUpdateReadingItems({
          item_ids: itemIds,
          action: "set_status",
          status
        })
        failed += response.failed
      }

      await fetchItems()
      if (failed > 0) {
        throw new Error(`Failed to restore ${failed} items`)
      }
    },
    [api, fetchItems]
  )

  const executeBulkDelete = useCallback(
    async (hardDelete: boolean) => {
      if (selectedItemIds.length === 0) {
        message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
        return
      }

      const previousStatusById = new Map<string, ReadingStatus>()
      selectedItemIds.forEach((itemId) => {
        const item = items.find((entry) => entry.id === itemId)
        previousStatusById.set(itemId, normalizeReadingStatus(item?.status))
      })

      setBulkActionLoading(true)
      try {
        const response = await api.bulkUpdateReadingItems({
          item_ids: selectedItemIds,
          action: "delete",
          hard: hardDelete
        })

        const succeededIds = new Set(
          response.results
            .filter((entry) => entry.success)
            .map((entry) => entry.item_id)
        )

        if (succeededIds.size > 0) {
          setSelectedItemIds((prev) => prev.filter((id) => !succeededIds.has(id)))
          await fetchItems()
        }

        const actionLabel = hardDelete
          ? t("collections:reading.bulk.actions.deletePermanent", "Delete permanently")
          : t("collections:reading.bulk.actions.delete", "Delete")

        if (response.failed > 0) {
          const failureLines = getBulkFailureLines(response)
          Modal.info({
            title: t("collections:reading.bulk.summaryTitle", "Bulk action summary"),
            width: 620,
            content: (
              <div className="space-y-2">
                <p>
                  {t(
                    "collections:reading.bulk.summaryBody",
                    "{{action}} completed. {{succeeded}} succeeded, {{failed}} failed.",
                    {
                      action: actionLabel,
                      succeeded: response.succeeded,
                      failed: response.failed
                    }
                  )}
                </p>
                {failureLines.length > 0 && (
                  <div className="max-h-52 overflow-auto rounded border border-border bg-bg p-2 text-xs">
                    {failureLines.map((line) => (
                      <div key={line}>{line}</div>
                    ))}
                    {response.failed > failureLines.length && (
                      <div>
                        {t("collections:reading.bulk.moreFailures", "+{{count}} more failures", {
                          count: response.failed - failureLines.length
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })
        }

        if (response.succeeded > 0 && hardDelete) {
          message.success(
            t(
              "collections:reading.bulk.deletePermanentSuccess",
              "Permanently deleted {{count}} items",
              { count: response.succeeded }
            )
          )
        }

        if (response.succeeded > 0 && !hardDelete) {
          const statusByItemId: Record<string, ReadingStatus> = {}
          response.results
            .filter((entry) => entry.success)
            .forEach((entry) => {
              statusByItemId[entry.item_id] = previousStatusById.get(entry.item_id) || "saved"
            })

          const restoredCount = Object.keys(statusByItemId).length
          showUndoNotification({
            title:
              restoredCount === 1
                ? t("collections:reading.bulk.deletedSingle", "Article deleted")
                : t("collections:reading.bulk.deletedMulti", "{{count}} articles deleted", {
                    count: restoredCount
                  }),
            description: t(
              "collections:reading.bulk.deleteUndoHint",
              "Moved to archived. Undo to restore previous statuses."
            ),
            onUndo: async () => {
              await restoreDeletedItems(statusByItemId)
            }
          })
        }
      } catch (error: any) {
        message.error(error?.message || "Bulk delete failed")
      } finally {
        setBulkActionLoading(false)
      }
    },
    [api, fetchItems, items, restoreDeletedItems, selectedItemIds, showUndoNotification, t]
  )

  const handleBulkDelete = useCallback(() => {
    if (selectedItemIds.length === 0) {
      message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
      return
    }
    Modal.confirm({
      title: t("collections:reading.bulk.deleteTitle", "Delete selected articles"),
      content: t(
        "collections:reading.bulk.deleteSoftBody",
        "Selected items will be moved to archived. You can undo this action."
      ),
      okText: t("common:delete", "Delete"),
      okButtonProps: { danger: true, loading: bulkActionLoading },
      cancelText: t("common:cancel", "Cancel"),
      onOk: async () => {
        await executeBulkDelete(false)
      }
    })
  }, [bulkActionLoading, executeBulkDelete, selectedItemIds.length, t])

  const handleBulkHardDelete = useCallback(() => {
    if (selectedItemIds.length === 0) {
      message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
      return
    }
    Modal.confirm({
      title: t("collections:reading.bulk.deletePermanentTitle", "Delete selected articles permanently"),
      content: t(
        "collections:reading.bulk.deletePermanentBody",
        "This permanently deletes selected items and cannot be undone. Continue?"
      ),
      okText: t("collections:reading.bulk.actions.deletePermanent", "Delete permanently"),
      okButtonProps: { danger: true, loading: bulkActionLoading },
      cancelText: t("common:cancel", "Cancel"),
      onOk: async () => {
        await executeBulkDelete(true)
      }
    })
  }, [bulkActionLoading, executeBulkDelete, selectedItemIds.length, t])

  const openBulkOutputModal = useCallback(async () => {
    if (selectedItemIds.length === 0) {
      message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
      return
    }
    setOutputModalOpen(true)
    setOutputTemplateId(null)
    setOutputTitle("")
    setOutputTemplatesLoading(true)
    try {
      const response = await api.getOutputTemplates({ limit: 200, offset: 0 })
      const options = (response.items || []).map((template: any) => ({
        label: `${template.name} (${String(template.format || "").toUpperCase()})`,
        value: String(template.id)
      }))
      setOutputTemplates(options)
      if (options.length === 1) {
        setOutputTemplateId(options[0].value)
      }
    } catch (error: any) {
      message.error(error?.message || "Failed to load templates")
      setOutputTemplates([])
    } finally {
      setOutputTemplatesLoading(false)
    }
  }, [api, selectedItemIds.length, t])

  const handleGenerateBulkOutput = useCallback(async () => {
    if (!outputTemplateId) {
      message.warning(t("collections:reading.bulk.templateRequired", "Select a template"))
      return
    }
    if (selectedItemIds.length === 0) {
      message.warning(t("collections:reading.bulk.selectWarning", "Select at least one item"))
      return
    }
    setOutputGenerating(true)
    try {
      const output = await api.generateOutput({
        template_id: outputTemplateId,
        item_ids: selectedItemIds,
        title: outputTitle.trim() || undefined
      })
      setOutputModalOpen(false)
      message.success(t("collections:reading.bulk.outputCreated", "Output generated"))
      Modal.success({
        title: t("collections:reading.bulk.outputSummaryTitle", "Output generation complete"),
        content: (
          <div className="space-y-1">
            <div>
              {t("collections:reading.bulk.outputSummaryBody", "Generated output #{{id}}.", {
                id: output.id
              })}
            </div>
            <div>
              {t("collections:reading.bulk.outputSummaryCount", "Items used: {{count}}", {
                count: selectedItemIds.length
              })}
            </div>
          </div>
        )
      })
    } catch (error: any) {
      message.error(error?.message || "Failed to generate output")
    } finally {
      setOutputGenerating(false)
    }
  }, [api, outputTemplateId, outputTitle, selectedItemIds, t])

  const bulkStatusMenuItems = useMemo<MenuProps["items"]>(() => {
    return STATUS_OPTIONS
      .filter((option) => option.value !== "all")
      .map((option) => ({
        key: option.value,
        label: t(`collections:status.${option.value}`, option.label),
        onClick: () => {
          void handleBulkStatus(option.value as ReadingStatus)
        }
      }))
  }, [handleBulkStatus, t])

  const bulkFavoriteMenuItems = useMemo<MenuProps["items"]>(() => {
    return [
      {
        key: "favorite",
        label: t("collections:reading.bulk.actions.favoriteOn", "Mark favorite"),
        onClick: () => {
          void handleBulkFavorite(true)
        }
      },
      {
        key: "unfavorite",
        label: t("collections:reading.bulk.actions.favoriteOff", "Unfavorite"),
        onClick: () => {
          void handleBulkFavorite(false)
        }
      }
    ]
  }, [handleBulkFavorite, t])

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Button
            type="primary"
            icon={<Plus className="h-4 w-4" />}
            onClick={openAddUrlModal}
          >
            {t("collections:reading.addUrl", "Add URL")}
          </Button>
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={fetchItems}
            loading={itemsLoading}
          >
            {t("common:refresh", "Refresh")}
          </Button>
          <Button onClick={handleSelectionModeToggle}>
            {selectionMode
              ? t("collections:reading.bulk.exitSelection", "Exit selection")
              : t("collections:reading.bulk.startSelection", "Select")}
          </Button>
        </div>

        <div className="flex flex-1 items-center gap-2 sm:max-w-md">
          <Input
            placeholder={t("collections:reading.searchPlaceholder", "Search articles...")}
            prefix={<Search className="h-4 w-4 text-text-subtle" />}
            value={itemsSearch}
            onChange={handleSearchChange}
            allowClear
          />
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-text-muted" />
          <span className="text-sm text-text-muted">
            {t("collections:reading.filters", "Filters")}:
          </span>
        </div>

        <Select
          value={filterStatus}
          onChange={setFilterStatus}
          options={STATUS_OPTIONS.map((opt) => ({
            ...opt,
            label: t(`collections:status.${opt.value}`, opt.label)
          }))}
          className="w-32"
          size="small"
        />

        <Button
          size="small"
          type={filterFavorite === true ? "primary" : "default"}
          icon={<Star className={`h-3 w-3 ${filterFavorite ? "fill-current" : ""}`} />}
          onClick={() => setFilterFavorite(filterFavorite === true ? null : true)}
        >
          {t("collections:reading.favorites", "Favorites")}
        </Button>

        <Select
          mode="tags"
          value={filterTags}
          onChange={handleTagFilterChange}
          options={availableTags.map((tag) => ({ value: tag, label: tag }))}
          className="min-w-40"
          size="small"
          placeholder={t("collections:reading.tagsFilter", "Tags")}
        />

        <Input
          value={filterDomain}
          onChange={handleDomainChange}
          placeholder={t("collections:reading.domainFilter", "Domain")}
          className="w-40"
          size="small"
          allowClear
        />

        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs text-text-muted">
            {t("collections:reading.dateFrom", "Date from")}
            <input
              aria-label={t("collections:reading.dateFrom", "Date from")}
              type="date"
              value={formatDateInputValue(filterDateFrom)}
              onChange={handleDateFromChange}
              className="h-6 rounded border border-border bg-bg px-2 text-xs text-text"
            />
          </label>
          <label className="flex items-center gap-1 text-xs text-text-muted">
            {t("collections:reading.dateTo", "Date to")}
            <input
              aria-label={t("collections:reading.dateTo", "Date to")}
              type="date"
              value={formatDateInputValue(filterDateTo)}
              onChange={handleDateToChange}
              className="h-6 rounded border border-border bg-bg px-2 text-xs text-text"
            />
          </label>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="text-sm text-text-muted">
            {t("collections:reading.sortBy", "Sort")}:
          </span>
          <Select
            value={sortBy}
            onChange={(v) => setSortBy(v as typeof sortBy)}
            options={SORT_OPTIONS.map((opt) => ({
              ...opt,
              label: t(`collections:sort.${opt.value}`, opt.label),
              disabled: opt.value === "relevance" && !itemsSearch
            }))}
            className="w-36"
            size="small"
          />
          <Button
            size="small"
            onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
          >
            {sortOrder === "asc" ? "↑" : "↓"}
          </Button>
        </div>

        {(filterStatus !== "all" ||
          filterFavorite !== null ||
          itemsSearch ||
          filterTags.length > 0 ||
          filterDomain ||
          filterDateFrom ||
          filterDateTo) && (
          <Button size="small" type="link" onClick={resetFilters}>
            {t("collections:reading.clearFilters", "Clear filters")}
          </Button>
        )}
      </div>

      {readingSavedSearchesEnabled && (
        <SavedSearchesMenu
          searches={savedSearches}
          loading={savedSearchesLoading}
          onApply={applySavedSearch}
          onCreateFromCurrent={() => void createSavedSearchFromCurrent()}
        />
      )}

      {/* Bulk action controls */}
      {selectionMode && (
        <div className="rounded-lg border border-border bg-bg p-3">
          <div className="flex flex-wrap items-center gap-3">
            <Checkbox
              checked={allPageSelected}
              indeterminate={selectedCount > 0 && !allPageSelected}
              onChange={handleSelectAllToggle}
            >
              {t("collections:reading.bulk.selectAllPage", "Select all on this page")}
            </Checkbox>
            <span className="text-sm text-text-muted">
              {t("collections:reading.bulk.selectedCount", "{{count}} selected", {
                count: selectedCount
              })}
            </span>
            <Button
              size="small"
              onClick={() => setSelectedItemIds([])}
              disabled={selectedCount === 0 || bulkActionLoading}
            >
              {t("collections:reading.bulk.clearSelection", "Clear selection")}
            </Button>
          </div>

          {selectedCount > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Dropdown menu={{ items: bulkStatusMenuItems }} trigger={["click"]}>
                <Button size="small" loading={bulkActionLoading}>
                  {t("collections:reading.bulk.actions.setStatus", "Set status")}
                </Button>
              </Dropdown>

              <Dropdown menu={{ items: bulkFavoriteMenuItems }} trigger={["click"]}>
                <Button size="small" loading={bulkActionLoading}>
                  {t("collections:reading.bulk.actions.favorite", "Favorite")}
                </Button>
              </Dropdown>

              <Button
                size="small"
                onClick={() => openTagActionModal("add")}
                loading={bulkActionLoading}
              >
                {t("collections:reading.bulk.actions.addTags", "Add tags")}
              </Button>

              <Button
                size="small"
                onClick={() => openTagActionModal("remove")}
                loading={bulkActionLoading}
              >
                {t("collections:reading.bulk.actions.removeTags", "Remove tags")}
              </Button>

              <Button
                size="small"
                onClick={openBulkOutputModal}
                loading={bulkActionLoading || outputGenerating}
              >
                {t("collections:reading.bulk.actions.generateOutput", "Generate output")}
              </Button>

              <Button
                danger
                size="small"
                icon={<Trash2 className="h-3 w-3" />}
                onClick={handleBulkDelete}
                loading={bulkActionLoading}
              >
                {t("collections:reading.bulk.actions.delete", "Delete")}
              </Button>

              <Button
                danger
                size="small"
                type="text"
                onClick={handleBulkHardDelete}
                loading={bulkActionLoading}
              >
                {t("collections:reading.bulk.actions.deletePermanent", "Delete permanently")}
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Items List */}
      {itemsLoading && items.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <Spin size="large" />
        </div>
      ) : itemsError ? (
        <Empty
          description={itemsError}
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        >
          <Button onClick={fetchItems}>{t("common:retry", "Retry")}</Button>
        </Empty>
      ) : items.length === 0 ? (
        <Empty
          description={
            itemsSearch || filterStatus !== "all" || filterFavorite !== null
              ? t("collections:reading.noResults", "No articles match your filters")
              : t("collections:reading.empty", "Your reading list is empty")
          }
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        >
          {!itemsSearch && filterStatus === "all" && filterFavorite === null && (
            <Button type="primary" onClick={openAddUrlModal}>
              {t("collections:reading.addFirst", "Add your first article")}
            </Button>
          )}
        </Empty>
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <ReadingItemCard
              key={item.id}
              item={item}
              onRefresh={fetchItems}
              selectionMode={selectionMode}
              selected={selectedItemIds.includes(item.id)}
              onSelectionChange={(checked) => handleItemSelectionChange(item.id, checked)}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {itemsTotal > itemsPageSize && (
        <div className="flex justify-center pt-4">
          <Pagination
            current={itemsPage}
            pageSize={itemsPageSize}
            total={itemsTotal}
            onChange={handlePageChange}
            showSizeChanger={false}
            showTotal={(total, range) =>
              t("collections:reading.pagination", "{{start}}-{{end}} of {{total}} items", {
                start: range[0],
                end: range[1],
                total
              })
            }
          />
        </div>
      )}

      {/* Modals */}
      {addUrlModalOpen && (
        <Suspense fallback={null}>
          <AddUrlModal onSuccess={fetchItems} />
        </Suspense>
      )}
      {itemDetailOpen && <ReadingItemDetail onRefresh={fetchItems} />}

      <Modal
        title={t("collections:reading.deleteConfirm.title", "Delete Article")}
        open={deleteConfirmOpen && deleteTargetType === "item"}
        onCancel={closeDeleteConfirm}
        onOk={handleDeleteConfirm}
        okText={t("common:delete", "Delete")}
        okButtonProps={{ danger: true, loading: deleteLoading }}
        cancelText={t("common:cancel", "Cancel")}
      >
        <p>
          {t(
            "collections:reading.deleteConfirm.message",
            "Are you sure you want to delete this article? This action cannot be undone."
          )}
        </p>
      </Modal>

      <Modal
        title={
          tagActionMode === "add"
            ? t("collections:reading.bulk.tagsModal.addTitle", "Add tags")
            : t("collections:reading.bulk.tagsModal.removeTitle", "Remove tags")
        }
        open={tagModalOpen}
        onCancel={() => setTagModalOpen(false)}
        onOk={handleTagActionConfirm}
        okText={
          tagActionMode === "add"
            ? t("collections:reading.bulk.actions.addTags", "Add tags")
            : t("collections:reading.bulk.actions.removeTags", "Remove tags")
        }
        okButtonProps={{ loading: bulkActionLoading }}
        cancelText={t("common:cancel", "Cancel")}
      >
        <Input
          value={tagInput}
          onChange={(e) => setTagInput(e.target.value)}
          placeholder={t("collections:reading.bulk.tagsModal.placeholder", "tag1, tag2, tag3")}
        />
      </Modal>

      <Modal
        title={t("collections:reading.bulk.outputTitle", "Generate output for selected items")}
        open={outputModalOpen}
        onCancel={() => setOutputModalOpen(false)}
        onOk={handleGenerateBulkOutput}
        okText={t("collections:reading.bulk.actions.generateOutput", "Generate output")}
        okButtonProps={{
          loading: outputGenerating,
          disabled: !outputTemplateId
        }}
        cancelText={t("common:cancel", "Cancel")}
      >
        <div className="space-y-3">
          <Select
            value={outputTemplateId || undefined}
            onChange={(value) => setOutputTemplateId(value)}
            options={outputTemplates}
            loading={outputTemplatesLoading}
            placeholder={t("collections:reading.bulk.templatePlaceholder", "Select a template")}
            className="w-full"
            showSearch
            optionFilterProp="label"
          />
          <Input
            value={outputTitle}
            onChange={(e) => setOutputTitle(e.target.value)}
            placeholder={t("collections:reading.bulk.outputNamePlaceholder", "Optional output title")}
          />
          <p className="text-xs text-text-muted">
            {t("collections:reading.bulk.outputHint", "Selected items: {{count}}", {
              count: selectedItemIds.length
            })}
          </p>
        </div>
      </Modal>
    </div>
  )
}
