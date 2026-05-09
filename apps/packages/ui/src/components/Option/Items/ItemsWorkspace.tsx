import React, { useCallback, useEffect, useMemo, useState } from "react"
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
  Tag,
  message
} from "antd"
import type { MenuProps } from "antd"
import { Filter, RefreshCw, Search, Star, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import {
  formatDateInputValue,
  formatDateOnlyLabel,
  parseDateInputValue
} from "@/utils/date-input"

type ItemStatus = "saved" | "reading" | "read" | "archived"

type SharedItem = {
  id: string
  content_item_id?: string
  media_id?: string
  title: string
  url?: string
  domain?: string
  summary?: string
  published_at?: string
  status?: ItemStatus
  favorite?: boolean
  tags: string[]
  type?: string
}

type SharedItemsBulkResponse = {
  total: number
  succeeded: number
  failed: number
  results: Array<{ item_id: string; success: boolean; error?: string | null }>
}

const STATUS_OPTIONS: Array<{ value: ItemStatus | "all"; label: string }> = [
  { value: "all", label: "All" },
  { value: "saved", label: "Saved" },
  { value: "reading", label: "Reading" },
  { value: "read", label: "Read" },
  { value: "archived", label: "Archived" }
]

const normalizeBulkTags = (raw: string): string[] => {
  if (!raw) return []
  const tags = raw
    .split(",")
    .map((entry) => entry.trim().toLowerCase())
    .filter((entry) => entry.length > 0)
  return Array.from(new Set(tags))
}

const getBulkFailureLines = (response: SharedItemsBulkResponse, maxLines = 10): string[] => {
  if (!Array.isArray(response.results) || maxLines <= 0) return []
  return response.results
    .filter((entry) => !entry.success)
    .slice(0, maxLines)
    .map((entry) => `#${entry.item_id}: ${entry.error || "update_failed"}`)
}

const normalizeItemStatus = (value: unknown): ItemStatus => {
  if (value === "saved" || value === "reading" || value === "read" || value === "archived") {
    return value
  }
  return "saved"
}

export const ItemsWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "collections", "common"])
  const api = useTldwApiClient()
  const { showUndoNotification } = useUndoNotification()

  const [items, setItems] = useState<SharedItem[]>([])
  const [itemsLoading, setItemsLoading] = useState(false)
  const [itemsError, setItemsError] = useState<string | null>(null)
  const [itemsTotal, setItemsTotal] = useState(0)
  const [itemsPage, setItemsPage] = useState(1)
  const [itemsPageSize] = useState(25)

  const [itemsSearch, setItemsSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<ItemStatus | "all">("all")
  const [favoriteFilter, setFavoriteFilter] = useState<boolean | null>(null)
  const [tagsFilter, setTagsFilter] = useState<string[]>([])
  const [domainFilter, setDomainFilter] = useState("")
  const [dateFrom, setDateFrom] = useState<string | null>(null)
  const [dateTo, setDateTo] = useState<string | null>(null)
  const [availableTags, setAvailableTags] = useState<string[]>([])

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

  const selectedCount = selectedItemIds.length
  const allPageSelected = items.length > 0 && selectedCount === items.length

  const fetchItems = useCallback(async () => {
    setItemsLoading(true)
    setItemsError(null)
    try {
      const response = await api.getItems({
        page: itemsPage,
        size: itemsPageSize,
        q: itemsSearch || undefined,
        status_filter: statusFilter !== "all" ? statusFilter : undefined,
        favorite: favoriteFilter ?? undefined,
        tags: tagsFilter.length > 0 ? tagsFilter : undefined,
        domain: domainFilter || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined
      })
      const nextItems = Array.isArray(response?.items) ? response.items : []
      setItems(nextItems)
      setItemsTotal(Number(response?.total || nextItems.length))
      const tagSet = new Set<string>()
      nextItems.forEach((item: SharedItem) => {
        item.tags?.forEach((tag) => tagSet.add(tag))
      })
      setAvailableTags(Array.from(tagSet).sort())
    } catch (error: any) {
      const msg = error?.message || "Failed to load items"
      setItemsError(msg)
      message.error(msg)
    } finally {
      setItemsLoading(false)
    }
  }, [
    api,
    itemsPage,
    itemsPageSize,
    itemsSearch,
    statusFilter,
    favoriteFilter,
    tagsFilter,
    domainFilter,
    dateFrom,
    dateTo
  ])

  useEffect(() => {
    void fetchItems()
  }, [fetchItems])

  useEffect(() => {
    const pageIds = new Set(items.map((item) => item.id))
    setSelectedItemIds((prev) => prev.filter((id) => pageIds.has(id)))
  }, [items])

  const applyBulkAction = useCallback(
    async (
      payload: {
        action: "set_status" | "set_favorite" | "add_tags" | "remove_tags" | "replace_tags" | "delete"
        status?: ItemStatus
        favorite?: boolean
        tags?: string[]
        hard?: boolean
      },
      actionLabel: string
    ): Promise<boolean> => {
      if (selectedItemIds.length === 0) {
        message.warning("Select at least one item")
        return false
      }

      setBulkActionLoading(true)
      try {
        const response = await api.bulkUpdateItems({
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
          const failureLines = getBulkFailureLines(response as SharedItemsBulkResponse)
          Modal.info({
            title: "Bulk action summary",
            width: 620,
            content: (
              <div className="space-y-2">
                <p>
                  {`${actionLabel} completed. ${response.succeeded} succeeded, ${response.failed} failed.`}
                </p>
                {failureLines.length > 0 && (
                  <div className="max-h-52 overflow-auto rounded border border-border bg-surface p-2 text-xs">
                    {failureLines.map((line) => (
                      <div key={line}>{line}</div>
                    ))}
                    {response.failed > failureLines.length && (
                      <div>{`+${response.failed - failureLines.length} more failures`}</div>
                    )}
                  </div>
                )}
              </div>
            )
          })
        } else {
          message.success(`${actionLabel} applied to ${response.succeeded} items`)
        }
        return true
      } catch (error: any) {
        message.error(error?.message || "Bulk action failed")
        return false
      } finally {
        setBulkActionLoading(false)
      }
    },
    [api, fetchItems, selectedItemIds]
  )

  const handleBulkStatus = useCallback(
    async (status: ItemStatus) => {
      await applyBulkAction(
        { action: "set_status", status },
        `Set status: ${status}`
      )
    },
    [applyBulkAction]
  )

  const handleBulkFavorite = useCallback(
    async (favorite: boolean) => {
      await applyBulkAction(
        { action: "set_favorite", favorite },
        favorite ? "Mark favorite" : "Unfavorite"
      )
    },
    [applyBulkAction]
  )

  const openTagActionModal = useCallback(
    (mode: "add" | "remove") => {
      if (selectedItemIds.length === 0) {
        message.warning("Select at least one item")
        return
      }
      setTagActionMode(mode)
      setTagInput("")
      setTagModalOpen(true)
    },
    [selectedItemIds.length]
  )

  const handleTagActionConfirm = useCallback(async () => {
    const tags = normalizeBulkTags(tagInput)
    if (tags.length === 0) {
      message.warning("Enter at least one tag")
      return
    }
    const action = tagActionMode === "add" ? "add_tags" : "remove_tags"
    const actionLabel = tagActionMode === "add" ? "Add tags" : "Remove tags"
    const ok = await applyBulkAction({ action, tags }, actionLabel)
    if (ok) {
      setTagModalOpen(false)
      setTagInput("")
    }
  }, [applyBulkAction, tagActionMode, tagInput])

  const restoreDeletedItems = useCallback(
    async (statusByItemId: Record<string, ItemStatus>) => {
      const grouped = new Map<ItemStatus, string[]>()
      Object.entries(statusByItemId).forEach(([itemId, status]) => {
        const normalizedStatus = normalizeItemStatus(status)
        const existing = grouped.get(normalizedStatus)
        if (existing) {
          existing.push(itemId)
        } else {
          grouped.set(normalizedStatus, [itemId])
        }
      })

      let failed = 0
      for (const [status, itemIds] of grouped.entries()) {
        const response = await api.bulkUpdateItems({
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
        message.warning("Select at least one item")
        return
      }

      const previousStatusById = new Map<string, ItemStatus>()
      selectedItemIds.forEach((itemId) => {
        const item = items.find((entry) => entry.id === itemId)
        previousStatusById.set(itemId, normalizeItemStatus(item?.status))
      })

      setBulkActionLoading(true)
      try {
        const response = await api.bulkUpdateItems({
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

        const actionLabel = hardDelete ? "Delete permanently" : "Delete"
        if (response.failed > 0) {
          const failureLines = getBulkFailureLines(response as SharedItemsBulkResponse)
          Modal.info({
            title: "Bulk action summary",
            width: 620,
            content: (
              <div className="space-y-2">
                <p>
                  {`${actionLabel} completed. ${response.succeeded} succeeded, ${response.failed} failed.`}
                </p>
                {failureLines.length > 0 && (
                  <div className="max-h-52 overflow-auto rounded border border-border bg-surface p-2 text-xs">
                    {failureLines.map((line) => (
                      <div key={line}>{line}</div>
                    ))}
                    {response.failed > failureLines.length && (
                      <div>{`+${response.failed - failureLines.length} more failures`}</div>
                    )}
                  </div>
                )}
              </div>
            )
          })
        }

        if (response.succeeded > 0 && hardDelete) {
          message.success(`Deleted ${response.succeeded} items permanently`)
        }

        if (response.succeeded > 0 && !hardDelete) {
          const statusByItemId: Record<string, ItemStatus> = {}
          response.results
            .filter((entry) => entry.success)
            .forEach((entry) => {
              statusByItemId[entry.item_id] = previousStatusById.get(entry.item_id) || "saved"
            })

          const restoredCount = Object.keys(statusByItemId).length
          showUndoNotification({
            title: restoredCount === 1 ? "Item deleted" : `${restoredCount} items deleted`,
            description: "Moved to archived. Undo to restore previous statuses.",
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
    [api, fetchItems, items, restoreDeletedItems, selectedItemIds, showUndoNotification]
  )

  const handleBulkDelete = useCallback(() => {
    if (selectedItemIds.length === 0) {
      message.warning("Select at least one item")
      return
    }
    Modal.confirm({
      title: "Delete selected items",
      content: "Selected items will be moved to archived. You can undo this action.",
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
      message.warning("Select at least one item")
      return
    }
    Modal.confirm({
      title: "Delete selected items permanently",
      content: "This permanently deletes selected items and cannot be undone. Continue?",
      okText: "Delete permanently",
      okButtonProps: { danger: true, loading: bulkActionLoading },
      cancelText: t("common:cancel", "Cancel"),
      onOk: async () => {
        await executeBulkDelete(true)
      }
    })
  }, [bulkActionLoading, executeBulkDelete, selectedItemIds.length, t])

  const openBulkOutputModal = useCallback(async () => {
    if (selectedItemIds.length === 0) {
      message.warning("Select at least one item")
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
  }, [api, selectedItemIds.length])

  const handleGenerateBulkOutput = useCallback(async () => {
    if (!outputTemplateId) {
      message.warning("Select a template")
      return
    }
    if (selectedItemIds.length === 0) {
      message.warning("Select at least one item")
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
      message.success("Output generated")
      Modal.success({
        title: "Output generation complete",
        content: (
          <div className="space-y-1">
            <div>{`Generated output #${output.id}.`}</div>
            <div>{`Items used: ${selectedItemIds.length}`}</div>
          </div>
        )
      })
    } catch (error: any) {
      message.error(error?.message || "Failed to generate output")
    } finally {
      setOutputGenerating(false)
    }
  }, [api, outputTemplateId, outputTitle, selectedItemIds])

  const bulkStatusMenuItems = useMemo<MenuProps["items"]>(() => {
    return STATUS_OPTIONS
      .filter((option) => option.value !== "all")
      .map((option) => ({
        key: option.value,
        label: option.label,
        onClick: () => {
          void handleBulkStatus(option.value as ItemStatus)
        }
      }))
  }, [handleBulkStatus])

  const bulkFavoriteMenuItems = useMemo<MenuProps["items"]>(() => {
    return [
      {
        key: "favorite",
        label: "Mark favorite",
        onClick: () => {
          void handleBulkFavorite(true)
        }
      },
      {
        key: "unfavorite",
        label: "Unfavorite",
        onClick: () => {
          void handleBulkFavorite(false)
        }
      }
    ]
  }, [handleBulkFavorite])

  const hasFilters =
    statusFilter !== "all" ||
    favoriteFilter !== null ||
    itemsSearch.trim().length > 0 ||
    tagsFilter.length > 0 ||
    domainFilter.trim().length > 0 ||
    Boolean(dateFrom) ||
    Boolean(dateTo)

  const resetFilters = useCallback(() => {
    setItemsSearch("")
    setStatusFilter("all")
    setFavoriteFilter(null)
    setTagsFilter([])
    setDomainFilter("")
    setDateFrom(null)
    setDateTo(null)
    setItemsPage(1)
  }, [])

  return (
    <WorkspaceConnectionGate
      featureName={t("collections:itemsTitle", "Items")}
      setupDescription={t(
        "collections:itemsSetupRequired",
        "Items depends on your connected tldw server to load, update, and generate outputs from saved reading items."
      )}
      maxWidthClassName="max-w-6xl"
    >
      <PageShell className="py-6 space-y-4" maxWidthClassName="max-w-6xl">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-text">Items</h1>
          <p className="text-sm text-text-muted">
            Manage shared items with bulk actions and output generation.
          </p>
        </div>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={fetchItems}
            loading={itemsLoading}
          >
            {t("common:refresh", "Refresh")}
          </Button>
          <Button
            onClick={() => {
              if (selectionMode) {
                setSelectionMode(false)
                setSelectedItemIds([])
              } else {
                setSelectionMode(true)
              }
            }}
          >
            {selectionMode ? "Exit selection" : "Select"}
          </Button>
        </div>

        <div className="flex flex-1 items-center gap-2 sm:max-w-md">
          <Input
            placeholder="Search items..."
            prefix={<Search className="h-4 w-4 text-text-subtle" />}
            value={itemsSearch}
            onChange={(e) => {
              setItemsSearch(e.target.value)
              setItemsPage(1)
            }}
            allowClear
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-text-muted" />
          <span className="text-sm text-text-muted">Filters:</span>
        </div>

        <Select
          value={statusFilter}
          onChange={(value) => {
            setStatusFilter(value as ItemStatus | "all")
            setItemsPage(1)
          }}
          options={STATUS_OPTIONS}
          className="w-32"
          size="small"
        />

        <Button
          size="small"
          type={favoriteFilter === true ? "primary" : "default"}
          icon={<Star className={`h-3 w-3 ${favoriteFilter ? "fill-current" : ""}`} />}
          onClick={() => {
            setFavoriteFilter(favoriteFilter === true ? null : true)
            setItemsPage(1)
          }}
        >
          Favorites
        </Button>

        <Select
          mode="tags"
          value={tagsFilter}
          onChange={(tags: string[]) => {
            const normalized = tags
              .map((tag) => tag.trim().toLowerCase())
              .filter(Boolean)
            setTagsFilter(Array.from(new Set(normalized)))
            setItemsPage(1)
          }}
          options={availableTags.map((tag) => ({ value: tag, label: tag }))}
          className="min-w-40"
          size="small"
          placeholder="Tags"
        />

        <Input
          value={domainFilter}
          onChange={(e) => {
            setDomainFilter(e.target.value)
            setItemsPage(1)
          }}
          placeholder="Domain"
          className="w-40"
          size="small"
          allowClear
        />

        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs text-text-muted">
            Date from
            <input
              aria-label="Date from"
              type="date"
              value={formatDateInputValue(dateFrom)}
              onChange={(event) => {
                setDateFrom(parseDateInputValue(event.target.value, "start"))
                setItemsPage(1)
              }}
              className="h-6 rounded border border-border bg-bg px-2 text-xs text-text"
            />
          </label>
          <label className="flex items-center gap-1 text-xs text-text-muted">
            Date to
            <input
              aria-label="Date to"
              type="date"
              value={formatDateInputValue(dateTo)}
              onChange={(event) => {
                setDateTo(parseDateInputValue(event.target.value, "end"))
                setItemsPage(1)
              }}
              className="h-6 rounded border border-border bg-bg px-2 text-xs text-text"
            />
          </label>
        </div>

        {hasFilters && (
          <Button size="small" type="link" onClick={resetFilters}>
            Clear filters
          </Button>
        )}
      </div>

      {selectionMode && (
        <div className="rounded-lg border border-border bg-surface p-3">
          <div className="flex flex-wrap items-center gap-3">
            <Checkbox
              checked={allPageSelected}
              indeterminate={selectedCount > 0 && !allPageSelected}
              onChange={() => {
                if (allPageSelected) {
                  setSelectedItemIds([])
                  return
                }
                setSelectedItemIds(items.map((item) => item.id))
              }}
            >
              Select all on this page
            </Checkbox>
            <span className="text-sm text-text-muted">
              {`${selectedCount} selected`}
            </span>
            <Button
              size="small"
              onClick={() => setSelectedItemIds([])}
              disabled={selectedCount === 0 || bulkActionLoading}
            >
              Clear selection
            </Button>
          </div>

          {selectedCount > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Dropdown menu={{ items: bulkStatusMenuItems }} trigger={["click"]}>
                <Button size="small" loading={bulkActionLoading}>
                  Set status
                </Button>
              </Dropdown>

              <Dropdown menu={{ items: bulkFavoriteMenuItems }} trigger={["click"]}>
                <Button size="small" loading={bulkActionLoading}>
                  Favorite
                </Button>
              </Dropdown>

              <Button
                size="small"
                onClick={() => openTagActionModal("add")}
                loading={bulkActionLoading}
              >
                Add tags
              </Button>

              <Button
                size="small"
                onClick={() => openTagActionModal("remove")}
                loading={bulkActionLoading}
              >
                Remove tags
              </Button>

              <Button
                size="small"
                onClick={openBulkOutputModal}
                loading={bulkActionLoading || outputGenerating}
              >
                Generate output
              </Button>

              <Button
                danger
                size="small"
                icon={<Trash2 className="h-3 w-3" />}
                onClick={handleBulkDelete}
                loading={bulkActionLoading}
              >
                Delete
              </Button>

              <Button
                danger
                size="small"
                type="text"
                onClick={handleBulkHardDelete}
                loading={bulkActionLoading}
              >
                Delete permanently
              </Button>
            </div>
          )}
        </div>
      )}

      {itemsLoading && items.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <Spin size="large" />
        </div>
      ) : itemsError ? (
        <Empty description={itemsError} image={Empty.PRESENTED_IMAGE_SIMPLE}>
          <Button onClick={fetchItems}>Retry</Button>
        </Empty>
      ) : items.length === 0 ? (
        <Empty
          description={hasFilters ? "No items match your filters" : "No items available"}
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <div className="space-y-3">
          {items.map((item) => {
            const selected = selectedItemIds.includes(item.id)
            const publishedLabel = formatDateOnlyLabel(item.published_at)

            return (
              <div
                key={item.id}
                className="rounded-lg border border-border bg-surface p-4"
              >
                <div className="flex items-start gap-3">
                  {selectionMode && (
                    <Checkbox
                      checked={selected}
                      onChange={(e) => {
                        const checked = e.target.checked
                        setSelectedItemIds((prev) => {
                          if (checked) {
                            if (prev.includes(item.id)) return prev
                            return [...prev, item.id]
                          }
                          return prev.filter((id) => id !== item.id)
                        })
                      }}
                      aria-label={`Toggle selection for ${item.title}`}
                    />
                  )}
                  <div className="min-w-0 flex-1 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-base font-medium text-text">
                        {item.title || "Untitled"}
                      </h3>
                      {item.type && (
                        <Tag color="blue">{item.type}</Tag>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
                      {item.domain && <span>{item.domain}</span>}
                      {publishedLabel && <span>{publishedLabel}</span>}
                      {item.url && (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          className="underline hover:text-primary"
                        >
                          Open source
                        </a>
                      )}
                    </div>

                    <p className="line-clamp-3 text-sm text-text">
                      {item.summary || "No summary available."}
                    </p>

                    {item.tags.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {item.tags.map((tag) => (
                          <Tag key={`${item.id}-${tag}`}>{tag}</Tag>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {itemsTotal > itemsPageSize && (
        <div className="flex justify-center pt-4">
          <Pagination
            current={itemsPage}
            pageSize={itemsPageSize}
            total={itemsTotal}
            onChange={setItemsPage}
            showSizeChanger={false}
            showTotal={(total, range) => `${range[0]}-${range[1]} of ${total} items`}
          />
        </div>
      )}

      <Modal
        title={tagActionMode === "add" ? "Add tags" : "Remove tags"}
        open={tagModalOpen}
        onCancel={() => setTagModalOpen(false)}
        onOk={handleTagActionConfirm}
        okText={tagActionMode === "add" ? "Add tags" : "Remove tags"}
        okButtonProps={{ loading: bulkActionLoading }}
        cancelText={t("common:cancel", "Cancel")}
      >
        <Input
          value={tagInput}
          onChange={(e) => setTagInput(e.target.value)}
          placeholder="tag1, tag2, tag3"
        />
      </Modal>

      <Modal
        title="Generate output for selected items"
        open={outputModalOpen}
        onCancel={() => setOutputModalOpen(false)}
        onOk={handleGenerateBulkOutput}
        okText="Generate output"
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
            placeholder="Select a template"
            className="w-full"
            showSearch
            optionFilterProp="label"
          />
          <Input
            value={outputTitle}
            onChange={(e) => setOutputTitle(e.target.value)}
            placeholder="Optional output title"
          />
          <p className="text-xs text-text-muted">
            {`Selected items: ${selectedItemIds.length}`}
          </p>
        </div>
      </Modal>
      </PageShell>
    </WorkspaceConnectionGate>
  )
}
