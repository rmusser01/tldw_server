import React, { useCallback, useEffect, useMemo, useState } from "react"
import {
  Alert,
  Button,
  Empty,
  Input,
  Modal,
  Pagination,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import {
  AlertTriangle,
  CheckCircle2,
  Circle,
  Copy,
  Download,
  Edit2,
  ExternalLink,
  Eye,
  LoaderCircle,
  Plus,
  RefreshCw,
  Trash2,
  UploadCloud,
  XCircle
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { useFetchWatchlistsQuery, useInvalidateWatchlists, useIsCurrentWatchlistsAccount, useWatchlistsQuery } from "@/hooks/useWatchlistsQuery"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  checkWatchlistSourcesNow,
  createWatchlistSource,
  deleteWatchlistSource,
  exportOpml,
  fetchWatchlistJobs,
  getSourceSeenStats,
  fetchWatchlistSources,
  fetchWatchlistGroups,
  fetchWatchlistTags,
  getWatchlistSettings,
  restoreWatchlistSource,
  updateWatchlistSource
} from "@/services/watchlists"
import type {
  SourceSeenStats,
  WatchlistGroup,
  WatchlistJob,
  WatchlistSource,
  WatchlistSourceCreate,
  WatchlistTag,
  SourceType
} from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { safeExternalUrl } from "@/utils/safe-external-url"
import { SourceFormModal } from "./SourceFormModal"
import { GroupsTree } from "./GroupsTree"
import { SourcesBulkImport } from "./SourcesBulkImport"
import { SourceSeenDrawer } from "./SourceSeenDrawer"
import {
  getSourcesTableEmptyDescription,
  shouldShowUnifiedWatchlistsEmptyState
} from "./empty-state"
import {
  normalizeSourceIds,
  shouldConfirmMultiSourceCheck
} from "./check-now-utils"
import { countToggleImpact, summarizeSourceSelection } from "./bulk-action-summary"
import {
  restoreDeletedSources,
  resolveBulkSourceUndoWindow,
  resolveSourceUndoWindowSeconds,
  SOURCE_DELETE_UNDO_WINDOW_SECONDS,
  toSourceRestoreId
} from "./source-undo"
import { getSourceStatusVisual, type SourceStatusIconToken } from "./sourceStatus"
import { findActiveSourceUsage, mapActiveSourceUsage } from "./source-usage"
import { mapWatchlistsError } from "../shared/watchlists-error"
import { useWatchlistsViewport } from "../shared/useWatchlistsViewport"
import { buildClonedWatchlistSourcePayload } from "./clone-utils"

const { Search } = Input

const SOURCE_TYPE_COLORS: Record<SourceType, string> = {
  rss: "blue",
  site: "green",
  forum: "purple"
}
type SourceHealthSnapshot = Pick<SourceSeenStats, "defer_until" | "consec_not_modified">
const EMPTY_SOURCES: WatchlistSource[] = []
const EMPTY_GROUPS: WatchlistGroup[] = []
const EMPTY_TAGS: WatchlistTag[] = []
const CATALOG_PARAMS = { page: 1, size: 200 }
const CLIENT_FILTER_PAGE_SIZE = 200
const CLIENT_FILTER_MAX_ITEMS = 1000
const SOURCE_USAGE_LOOKUP_PAGE_SIZE = 200
const SOURCE_USAGE_LOOKUP_MAX_PAGES = 10
const BULK_MOVE_NO_GROUP_VALUE = "__none__"
const SOURCES_ADVANCED_COLUMNS_STORAGE_KEY = "watchlists:sources:advanced-columns:v1"

const readStoredDisclosureState = (key: string): boolean | null => {
  if (typeof window === "undefined") return null
  try {
    const raw = window.localStorage.getItem(key)
    if (raw === "1") return true
    if (raw === "0") return false
  } catch {
    return null
  }
  return null
}

const persistDisclosureState = (key: string, value: boolean): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, value ? "1" : "0")
  } catch {
    // Ignore storage persistence failures
  }
}

const renderSourceStatusIcon = (iconToken: SourceStatusIconToken) => {
  if (iconToken === "healthy") return <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
  if (iconToken === "activity") return <LoaderCircle className="h-3.5 w-3.5" aria-hidden />
  if (iconToken === "attention") return <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
  if (iconToken === "error") return <XCircle className="h-3.5 w-3.5" aria-hidden />
  return <Circle className="h-3.5 w-3.5" aria-hidden />
}

type BulkMoveTargetValue = number | typeof BULK_MOVE_NO_GROUP_VALUE | null

const toNormalizedGroupIds = (source: WatchlistSource): number[] => {
  if (!Array.isArray(source.group_ids)) return []
  return source.group_ids
    .map((groupId) => Number(groupId))
    .filter((groupId) => Number.isFinite(groupId))
}

export const SourcesTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const { showUndoNotification } = useUndoNotification()
  const { isConstrained } = useWatchlistsViewport()

  // Store state
  const sourcesSearch = useWatchlistsStore((s) => s.sourcesSearch)
  const sourcesPage = useWatchlistsStore((s) => s.sourcesPage)
  const sourcesPageSize = useWatchlistsStore((s) => s.sourcesPageSize)
  const selectedGroupId = useWatchlistsStore((s) => s.selectedGroupId)
  const selectedTagName = useWatchlistsStore((s) => s.selectedTagName)
  const sourceFormOpen = useWatchlistsStore((s) => s.sourceFormOpen)
  const sourceFormEditId = useWatchlistsStore((s) => s.sourceFormEditId)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)

  // Store actions
  const setSourcesSearch = useWatchlistsStore((s) => s.setSourcesSearch)
  const setSourcesPage = useWatchlistsStore((s) => s.setSourcesPage)
  const setSourcesPageSize = useWatchlistsStore((s) => s.setSourcesPageSize)
  const setActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const setSelectedGroupId = useWatchlistsStore((s) => s.setSelectedGroupId)
  const setSelectedTagName = useWatchlistsStore((s) => s.setSelectedTagName)
  const openSourceForm = useWatchlistsStore((s) => s.openSourceForm)
  const closeSourceForm = useWatchlistsStore((s) => s.closeSourceForm)

  // Local state
  const [selectedTypeFilter, setSelectedTypeFilter] = useState<string | null>(
    null
  )
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])
  const [bulkWorking, setBulkWorking] = useState(false)
  const [sourceCloneDraft, setSourceCloneDraft] = useState<WatchlistSourceCreate | null>(null)
  const [bulkMoveTargetValue, setBulkMoveTargetValue] = useState<BulkMoveTargetValue>(null)
  const [checkingSourceIds, setCheckingSourceIds] = useState<number[]>([])
  const [importOpen, setImportOpen] = useState(false)
  const [seenDrawerSourceId, setSeenDrawerSourceId] = useState<number | null>(null)
  const [showAdvancedColumns, setShowAdvancedColumns] = useState<boolean>(() => {
    const stored = readStoredDisclosureState(SOURCES_ADVANCED_COLUMNS_STORAGE_KEY)
    return stored ?? false
  })
  const invalidateWatchlists = useInvalidateWatchlists()
  const isCurrentAccount = useIsCurrentWatchlistsAccount()
  const fetchWatchlistsQuery = useFetchWatchlistsQuery()
  const refreshSourceCatalog = useCallback(
    () => invalidateWatchlists("sources", "source-seen", "groups", "tags", "job-preview", "overview"),
    [invalidateWatchlists]
  )
  const sourceFilters = {
    watchlist_id: selectedWatchlistId ?? undefined,
    q: sourcesSearch || undefined,
    tags: selectedTagName ? [selectedTagName] : undefined,
    groups: selectedGroupId != null ? [selectedGroupId] : undefined
  }
  const sourceParams = { ...sourceFilters, page: sourcesPage, size: sourcesPageSize }
  const sourcesQuery = useWatchlistsQuery(
    "sources",
    selectedTypeFilter ? { ...sourceParams, sourceType: selectedTypeFilter } : sourceParams,
    async ({ signal }) => {
      if (!selectedTypeFilter) return fetchWatchlistSources(sourceParams, signal)

      const items: WatchlistSource[] = []
      for (let page = 1; items.length < CLIENT_FILTER_MAX_ITEMS; page += 1) {
        const result = await fetchWatchlistSources({
          ...sourceFilters,
          page,
          size: CLIENT_FILTER_PAGE_SIZE
        }, signal)
        const batch = Array.isArray(result.items) ? result.items : []
        items.push(...batch)
        if (items.length >= result.total || batch.length < CLIENT_FILTER_PAGE_SIZE) break
      }
      const matching = items.filter((source) => source.source_type === selectedTypeFilter)
      return {
        items: matching.slice((sourcesPage - 1) * sourcesPageSize, sourcesPage * sourcesPageSize),
        total: matching.length
      }
    }
  )
  const sources = Array.isArray(sourcesQuery.data?.items) ? sourcesQuery.data.items : EMPTY_SOURCES
  const sourcesLoading = sourcesQuery.isFetching
  const sourcesTotal = sourcesQuery.data?.total ?? sources.length
  const sourcesLoadError = sourcesQuery.error
    ? mapWatchlistsError(sourcesQuery.error, {
        t,
        context: t("watchlists:sources.title", "Feeds"),
        fallbackMessage: t("watchlists:sources.fetchError", "Failed to load feeds")
      })
    : null
  const refreshSources = () => void invalidateWatchlists("sources", "source-seen")

  const groupsQuery = useWatchlistsQuery("groups", CATALOG_PARAMS, ({ signal }) => fetchWatchlistGroups(CATALOG_PARAMS, signal))
  const groups = Array.isArray(groupsQuery.data?.items) ? groupsQuery.data.items : EMPTY_GROUPS
  const groupsLoading = groupsQuery.isFetching
  const { data: tagsPage } = useWatchlistsQuery("tags", CATALOG_PARAMS, ({ signal }) => fetchWatchlistTags(CATALOG_PARAMS, signal))
  const tags = Array.isArray(tagsPage?.items) ? tagsPage.items : EMPTY_TAGS
  const { data: settings } = useWatchlistsQuery("settings", {}, ({ signal }) => getWatchlistSettings(signal))

  const sourceIds = sources.map((source) => source.id)
  const healthParams = { keys_limit: 1 }
  const { data: sourceHealthById } = useWatchlistsQuery(
    "source-seen",
    { sourceIds, ...healthParams },
    async ({ signal }) => {
      const results = await Promise.allSettled(
        sourceIds.map((id) => getSourceSeenStats(id, healthParams, signal))
      )
      const health: Record<number, SourceHealthSnapshot> = {}
      results.forEach((result, index) => {
        if (result.status !== "fulfilled") return
        health[sourceIds[index]] = {
          defer_until: result.value.defer_until,
          consec_not_modified: result.value.consec_not_modified
        }
      })
      return health
    },
    { enabled: sourceIds.length > 0 }
  )

  const selectedSources = useMemo(
    () => sources.filter((source) => selectedRowKeys.includes(source.id)),
    [sources, selectedRowKeys]
  )
  const selectedSourceIds = useMemo(
    () => selectedSources.map((source) => source.id),
    [selectedSources]
  )
  const selectedSourceSummary = useMemo(
    () => summarizeSourceSelection(selectedSources),
    [selectedSources]
  )
  const hasActiveFilters = useMemo(
    () =>
      Boolean(
        selectedGroupId ||
          selectedTagName ||
          selectedTypeFilter ||
          sourcesSearch.trim().length > 0
      ),
    [selectedGroupId, selectedTagName, selectedTypeFilter, sourcesSearch]
  )
  const unifiedEmptyState = shouldShowUnifiedWatchlistsEmptyState({
    groupsCount: groups.length,
    sourcesCount: sources.length,
    hasActiveFilters,
    groupsLoading,
    sourcesLoading
  })
  const tableEmptyDescription = getSourcesTableEmptyDescription(hasActiveFilters)

  useEffect(() => {
    if (selectedRowKeys.length === 0 && bulkMoveTargetValue !== null) {
      setBulkMoveTargetValue(null)
    }
  }, [bulkMoveTargetValue, selectedRowKeys.length])

  useEffect(() => {
    persistDisclosureState(SOURCES_ADVANCED_COLUMNS_STORAGE_KEY, showAdvancedColumns)
  }, [showAdvancedColumns])

  // Handle toggle active
  const handleToggleActive = async (source: WatchlistSource) => {
    try {
      await updateWatchlistSource(source.id, {
        active: !source.active
      })
      void invalidateWatchlists("sources", "job-preview", "overview")
      message.success(
        source.active
          ? t("watchlists:sources.disabled", "Source disabled")
          : t("watchlists:sources.enabled", "Source enabled")
      )
    } catch (err) {
      console.error("Failed to toggle source:", err)
      message.error(t("watchlists:sources.toggleError", "Failed to update source"))
    }
  }

  const loadJobsForSourceUsageCheck = useCallback(() => {
    const params = {
      watchlist_id: selectedWatchlistId ?? undefined,
      size: SOURCE_USAGE_LOOKUP_PAGE_SIZE
    }
    return fetchWatchlistsQuery("jobs", { ...params, maxPages: SOURCE_USAGE_LOOKUP_MAX_PAGES }, async ({ signal }) => {
      const allJobs: WatchlistJob[] = []
      for (let page = 1; page <= SOURCE_USAGE_LOOKUP_MAX_PAGES; page += 1) {
        const response = await fetchWatchlistJobs({ ...params, page }, signal)
        const pageItems = Array.isArray(response.items) ? response.items : []
        allJobs.push(...pageItems)
        if (pageItems.length < params.size || allJobs.length >= Number(response.total || 0)) break
      }
      return allJobs
    })
  }, [fetchWatchlistsQuery, selectedWatchlistId])

  // Handle delete
  const handleDelete = async (sourceId: number) => {
    if (!isCurrentAccount()) return
    const deletedSource = sources.find((source) => source.id === sourceId)
    try {
      const deleteResult = await deleteWatchlistSource(sourceId)
      if (!isCurrentAccount()) return
      void refreshSourceCatalog()
      if (!deletedSource) {
        message.success(t("watchlists:sources.deleted", "Feed deleted"))
        return
      }
      const undoWindowSeconds = resolveSourceUndoWindowSeconds(
        deleteResult?.restore_window_seconds,
        SOURCE_DELETE_UNDO_WINDOW_SECONDS
      )

      showUndoNotification({
        title: t("watchlists:sources.undoDeleteTitle", "Feed deleted"),
        description: t(
          "watchlists:sources.undoDeleteDescription",
          "Undo restores this feed's tags, groups, and active state for {{seconds}} seconds.",
          { seconds: undoWindowSeconds }
        ),
        duration: undoWindowSeconds,
        onUndo: async () => {
          if (!isCurrentAccount()) throw new DOMException("Watchlists account changed", "AbortError")
          try {
            await restoreWatchlistSource(toSourceRestoreId(deletedSource))
          } catch (restoreErr) {
            const restoreError = mapWatchlistsError(restoreErr, {
              t,
              context: t("watchlists:sources.title", "Feeds"),
              operationLabel: t("watchlists:errors.operation.retry", "retry"),
              fallbackMessage: t(
                "watchlists:sources.undoRestoreError",
                "Could not restore this feed. Refresh Feeds and retry while the undo timer is active."
              )
            })
            throw new Error(restoreError.description)
          } finally {
            await refreshSourceCatalog()
          }
        }
      })
    } catch (err) {
      console.error("Failed to delete source:", err)
      message.error(t("watchlists:sources.deleteError", "Failed to delete source"))
    }
  }

  const requestDeleteConfirmation = useCallback(async (source: WatchlistSource) => {
    let activeUsage: Array<{ id: number; name: string }> = []
    try {
      const jobs = await loadJobsForSourceUsageCheck()
      activeUsage = findActiveSourceUsage(jobs, source.id)
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      console.error("Failed to check source usage:", err)
      message.warning(
        t(
          "watchlists:sources.deleteUsageLookupError",
          "Could not verify active monitor usage before deletion."
        )
      )
    }

    if (!isCurrentAccount()) return
    const usageCount = activeUsage.length
    Modal.confirm({
      title: usageCount > 0
        ? t("watchlists:sources.deleteConfirmInUseTitle", "Feed is used by active monitors")
        : t("watchlists:sources.deleteConfirm", "Delete this feed?"),
      content: usageCount > 0 ? (
        <div className="space-y-2">
          <p>
            {t(
              "watchlists:sources.deleteConfirmInUseDescription",
              "This feed is referenced by {{count}} active monitor{{plural}}. Deleting it may break scheduled runs.",
              {
                count: usageCount,
                plural: usageCount === 1 ? "" : "s"
              }
            )}
          </p>
          <p className="text-sm text-text-muted">
            {t(
              "watchlists:sources.deleteConfirmDescription",
              "You can undo this deletion for {{seconds}} seconds.",
              { seconds: SOURCE_DELETE_UNDO_WINDOW_SECONDS }
            )}
          </p>
          <ul className="list-disc pl-5">
            {activeUsage.slice(0, 5).map((job) => (
              <li key={job.id}>{job.name}</li>
            ))}
          </ul>
          {usageCount > 5 && (
            <p className="text-xs text-text-muted">
              {t(
                "watchlists:sources.deleteConfirmInUseMore",
                "+{{count}} more active monitor{{plural}}",
                {
                  count: usageCount - 5,
                  plural: usageCount - 5 === 1 ? "" : "s"
                }
              )}
            </p>
          )}
        </div>
      ) : (
        t(
          "watchlists:sources.deleteConfirmDescription",
          "You can undo this deletion for {{seconds}} seconds.",
          { seconds: SOURCE_DELETE_UNDO_WINDOW_SECONDS }
        )
      ),
      okText: usageCount > 0
        ? t("watchlists:sources.deleteConfirmForce", "Delete anyway")
        : t("watchlists:sources.delete", "Delete"),
      cancelText: t("common:cancel", "Cancel"),
      okButtonProps: { danger: true },
      onOk: () => handleDelete(source.id)
    })
  }, [handleDelete, isCurrentAccount, loadJobsForSourceUsageCheck, t])

  const handleBulkToggle = async (active: boolean) => {
    if (!isCurrentAccount() || selectedSources.length === 0) return
    const toggledSources = [...selectedSources]
    setBulkWorking(true)
    try {
      const results = await Promise.allSettled(
        toggledSources.map((source) =>
          updateWatchlistSource(source.id, { active })
        )
      )
      let successCount = 0
      const previouslyToggledSources: WatchlistSource[] = []
      results.forEach((result, idx) => {
        if (result.status === "fulfilled") {
          const source = toggledSources[idx]
          previouslyToggledSources.push(source)
          successCount += 1
        }
      })
      const failedCount = toggledSources.length - successCount

      if (successCount > 0) {
        void refreshSourceCatalog()
        showUndoNotification({
          title: t(
            "watchlists:sources.undoBulkToggleTitle",
            "Updated {{count}} feeds",
            { count: successCount }
          ),
          description: t(
            "watchlists:sources.undoBulkToggleDescription",
            "Undo restores previous feed states for {{seconds}} seconds.",
            { seconds: SOURCE_DELETE_UNDO_WINDOW_SECONDS }
          ),
          duration: SOURCE_DELETE_UNDO_WINDOW_SECONDS,
          onUndo: async () => {
            if (!isCurrentAccount()) throw new DOMException("Watchlists account changed", "AbortError")
            const restoreResults = await Promise.allSettled(
              previouslyToggledSources.map((source) =>
                updateWatchlistSource(source.id, { active: source.active })
              )
            )
            let restoredCount = 0
            let restoreFailures = 0
            restoreResults.forEach((result) => {
              if (result.status === "fulfilled") {
                restoredCount += 1
              } else {
                restoreFailures += 1
              }
            })

            await refreshSourceCatalog()
            if (restoreFailures > 0) {
              throw new Error(
                t(
                  "watchlists:sources.undoBulkTogglePartialError",
                  "{{restored}} restored, {{failed}} failed to restore",
                  {
                    restored: restoredCount,
                    failed: restoreFailures
                  }
                )
              )
            }
          }
        })
      }

      if (failedCount > 0) {
        message.warning(
          t(
            "watchlists:sources.bulkUpdatePartial",
            "Updated {{success}} feeds, {{failed}} failed",
            { success: successCount, failed: failedCount }
          )
        )
      }

      if (successCount === 0) {
        message.error(t("watchlists:sources.bulkError", "Bulk update failed"))
      }
    } catch (err) {
      console.error("Bulk update failed:", err)
      message.error(t("watchlists:sources.bulkError", "Bulk update failed"))
    } finally {
      setBulkWorking(false)
      setSelectedRowKeys([])
      setBulkMoveTargetValue(null)
    }
  }

  const handleBulkDelete = async () => {
    if (!isCurrentAccount() || selectedSources.length === 0) return
    const deletedSources = [...selectedSources]
    setBulkWorking(true)
    try {
      const results = await Promise.allSettled(
        deletedSources.map((source) => deleteWatchlistSource(source.id))
      )
      let successCount = 0
      const removedSources: WatchlistSource[] = []
      const successfulDeletePayloads: Array<{ restore_window_seconds?: number | string | null }> = []
      results.forEach((result, idx) => {
        if (result.status === "fulfilled") {
          const removedSource = deletedSources[idx]
          removedSources.push(removedSource)
          successfulDeletePayloads.push(result.value)
          successCount += 1
        }
      })
      const failedCount = deletedSources.length - successCount

      if (successCount > 0) {
        void refreshSourceCatalog()
        const undoWindow = resolveBulkSourceUndoWindow(
          successfulDeletePayloads,
          SOURCE_DELETE_UNDO_WINDOW_SECONDS
        )
        const undoDescription = t(
          "watchlists:sources.undoBulkDeleteDescription",
          "Undo restores deleted feeds, tags, groups, and active states for {{seconds}} seconds.",
          { seconds: undoWindow.seconds }
        )
        const variableWindowHint = undoWindow.hasMixedWindows
          ? ` ${t(
              "watchlists:sources.undoBulkDeleteVariableWindowHint",
              "Some feeds expire sooner; use Undo before this timer ends."
            )}`
          : ""

        showUndoNotification({
          title: t(
            "watchlists:sources.undoBulkDeleteTitle",
            "{{count}} feeds deleted",
            { count: successCount }
          ),
          description: `${undoDescription}${variableWindowHint}`,
          duration: undoWindow.seconds,
          onUndo: async () => {
            if (!isCurrentAccount()) throw new DOMException("Watchlists account changed", "AbortError")
            const restoreSummary = await restoreDeletedSources(
              removedSources,
              restoreWatchlistSource
            )
            await refreshSourceCatalog()
            if (restoreSummary.failed > 0) {
              throw new Error(
                t(
                  "watchlists:sources.undoBulkRestorePartialError",
                  "{{restored}} restored, {{failed}} failed to restore. Refresh Feeds and retry while the undo timer is active.",
                  {
                    restored: restoreSummary.restored,
                    failed: restoreSummary.failed
                  }
                )
              )
            }
          }
        })
      }

      if (failedCount > 0) {
        message.warning(
          t(
            "watchlists:sources.bulkDeletePartial",
            "Deleted {{success}} feeds, {{failed}} failed",
            { success: successCount, failed: failedCount }
          )
        )
      }

      if (successCount === 0) {
        message.error(t("watchlists:sources.bulkDeleteError", "Bulk delete failed"))
      }
    } catch (err) {
      console.error("Bulk delete failed:", err)
      message.error(t("watchlists:sources.bulkDeleteError", "Bulk delete failed"))
    } finally {
      setBulkWorking(false)
      setSelectedRowKeys([])
      setBulkMoveTargetValue(null)
    }
  }

  const requestBulkToggle = useCallback((active: boolean) => {
    if (selectedSources.length === 0) return
    const impacted = countToggleImpact(selectedSources, active)
    Modal.confirm({
      title: active
        ? t("watchlists:sources.bulkEnableConfirmTitle", "Enable selected feeds?")
        : t("watchlists:sources.bulkDisableConfirmTitle", "Disable selected feeds?"),
      content: t(
        "watchlists:sources.bulkToggleConfirmDescription",
        "{{total}} selected ({{active}} active, {{inactive}} inactive). {{impacted}} will change state.",
        {
          total: selectedSourceSummary.total,
          active: selectedSourceSummary.active,
          inactive: selectedSourceSummary.inactive,
          impacted
        }
      ),
      okText: active
        ? t("watchlists:sources.bulkEnable", "Enable")
        : t("watchlists:sources.bulkDisable", "Disable"),
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => handleBulkToggle(active)
    })
  }, [selectedSourceSummary, selectedSources, t])

  const requestBulkDelete = useCallback(async () => {
    if (!isCurrentAccount() || selectedSources.length === 0) return

    const selectedIds = selectedSources.map((source) => source.id)
    let usageMap = new Map<number, Array<{ id: number; name: string }>>()

    try {
      const jobs = await loadJobsForSourceUsageCheck()
      usageMap = mapActiveSourceUsage(jobs, selectedIds)
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return
      console.error("Failed to check selected source usage:", err)
      message.warning(
        t(
          "watchlists:sources.deleteUsageLookupError",
          "Could not verify active monitor usage before deletion."
        )
      )
    }

    if (!isCurrentAccount()) return
    const impactedSourcesCount = selectedIds.filter((id) => (usageMap.get(id)?.length ?? 0) > 0).length
    const impactedJobs = Array.from(
      new Map(
        selectedIds.flatMap((id) => (usageMap.get(id) || []).map((job) => [job.id, job]))
      ).values()
    )

    Modal.confirm({
      title: t(
        "watchlists:sources.bulkDeleteConfirmTitle",
        "Delete {{count}} selected feeds?",
        { count: selectedSourceSummary.total }
      ),
      content: (
        <div className="space-y-2">
          <p>
            {t(
              "watchlists:sources.bulkDeleteConfirmDescription",
              "This will delete {{count}} feeds ({{active}} active, {{inactive}} inactive). Undo is available for {{seconds}} seconds.",
              {
                count: selectedSourceSummary.total,
                active: selectedSourceSummary.active,
                inactive: selectedSourceSummary.inactive,
                seconds: SOURCE_DELETE_UNDO_WINDOW_SECONDS
              }
            )}
          </p>
          {impactedSourcesCount > 0 && (
            <>
              <p>
                {t(
                  "watchlists:sources.bulkDeleteConfirmInUseSummary",
                  "{{sources}} selected feed{{sourcesPlural}} are referenced by {{jobs}} active monitor{{jobsPlural}}.",
                  {
                    sources: impactedSourcesCount,
                    sourcesPlural: impactedSourcesCount === 1 ? "" : "s",
                    jobs: impactedJobs.length,
                    jobsPlural: impactedJobs.length === 1 ? "" : "s"
                  }
                )}
              </p>
              <ul className="list-disc pl-5">
                {impactedJobs.slice(0, 5).map((job) => (
                  <li key={job.id}>{job.name}</li>
                ))}
              </ul>
              {impactedJobs.length > 5 && (
                <p className="text-xs text-text-muted">
                  {t(
                    "watchlists:sources.bulkDeleteConfirmInUseMore",
                    "+{{count}} more active monitor{{plural}}",
                    {
                      count: impactedJobs.length - 5,
                      plural: impactedJobs.length - 5 === 1 ? "" : "s"
                    }
                  )}
                </p>
              )}
            </>
          )}
        </div>
      ),
      okText: t("watchlists:sources.bulkDelete", "Delete"),
      cancelText: t("common:cancel", "Cancel"),
      okButtonProps: { danger: true },
      onOk: () => handleBulkDelete()
    })
  }, [handleBulkDelete, isCurrentAccount, loadJobsForSourceUsageCheck, selectedSourceSummary, selectedSources, t])

  const handleBulkMoveToGroup = useCallback(async (targetGroupId: number | null) => {
    if (!isCurrentAccount() || selectedSources.length === 0) return
    const movedSources = [...selectedSources]
    const previousGroupIdsBySourceId = new Map<number, number[]>()
    movedSources.forEach((source) => {
      previousGroupIdsBySourceId.set(source.id, toNormalizedGroupIds(source))
    })

    setBulkWorking(true)
    try {
      const results = await Promise.allSettled(
        movedSources.map((source) =>
          updateWatchlistSource(source.id, {
            group_ids: targetGroupId == null ? [] : [targetGroupId]
          })
        )
      )

      const updatedSources: WatchlistSource[] = []
      let successCount = 0
      results.forEach((result, idx) => {
        if (result.status !== "fulfilled") return
        const source = movedSources[idx]
        updatedSources.push(source)
        successCount += 1
      })
      const failedCount = movedSources.length - successCount

      if (successCount > 0) {
        void refreshSourceCatalog()
        showUndoNotification({
          title: t("watchlists:sources.undoBulkMoveTitle", "Moved {{count}} feeds", {
            count: successCount
          }),
          description: t(
            "watchlists:sources.undoBulkMoveDescription",
            "Undo restores previous group assignments for {{seconds}} seconds.",
            { seconds: SOURCE_DELETE_UNDO_WINDOW_SECONDS }
          ),
          duration: SOURCE_DELETE_UNDO_WINDOW_SECONDS,
          onUndo: async () => {
            if (!isCurrentAccount()) throw new DOMException("Watchlists account changed", "AbortError")
            const restoreResults = await Promise.allSettled(
              updatedSources.map((source) =>
                updateWatchlistSource(source.id, {
                  group_ids: previousGroupIdsBySourceId.get(source.id) || []
                })
              )
            )
            let restoredCount = 0
            let restoreFailures = 0
            restoreResults.forEach((result) => {
              if (result.status === "fulfilled") {
                restoredCount += 1
              } else {
                restoreFailures += 1
              }
            })
            await refreshSourceCatalog()
            if (restoreFailures > 0) {
              throw new Error(
                t(
                  "watchlists:sources.undoBulkMovePartialError",
                  "{{restored}} restored, {{failed}} failed to restore",
                  {
                    restored: restoredCount,
                    failed: restoreFailures
                  }
                )
              )
            }
          }
        })
      }

      if (failedCount > 0) {
        message.warning(
          t(
            "watchlists:sources.bulkMovePartial",
            "Moved {{success}} feeds, {{failed}} failed",
            { success: successCount, failed: failedCount }
          )
        )
      }

      if (successCount === 0) {
        message.error(t("watchlists:sources.bulkMoveError", "Bulk move failed"))
      }
    } catch (err) {
      console.error("Failed to bulk move sources:", err)
      message.error(t("watchlists:sources.bulkMoveError", "Bulk move failed"))
    } finally {
      setBulkWorking(false)
      setSelectedRowKeys([])
      setBulkMoveTargetValue(null)
    }
  }, [isCurrentAccount, refreshSourceCatalog, selectedSources, showUndoNotification, t])

  const requestBulkMoveToGroup = useCallback(() => {
    if (selectedSources.length === 0 || bulkMoveTargetValue == null) return
    const parsedTargetGroupId = bulkMoveTargetValue === BULK_MOVE_NO_GROUP_VALUE
      ? null
      : Number(bulkMoveTargetValue)

    if (parsedTargetGroupId !== null && !Number.isFinite(parsedTargetGroupId)) {
      message.error(t("watchlists:sources.bulkMoveError", "Bulk move failed"))
      return
    }

    const targetGroupName = parsedTargetGroupId == null
      ? t("watchlists:sources.moveTargetNone", "No group")
      : groups.find((group) => group.id === parsedTargetGroupId)?.name ||
        t("watchlists:sources.moveUnknownGroup", "Selected group")

    Modal.confirm({
      title: t(
        "watchlists:sources.bulkMoveConfirmTitle",
        "Move {{count}} selected feeds?",
        { count: selectedSourceSummary.total }
      ),
      content: t(
        "watchlists:sources.bulkMoveConfirmDescription",
        "Feeds will be assigned to {{group}}.",
        { group: targetGroupName }
      ),
      okText: t("watchlists:sources.bulkMove", "Move"),
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => handleBulkMoveToGroup(parsedTargetGroupId)
    })
  }, [
    bulkMoveTargetValue,
    groups,
    handleBulkMoveToGroup,
    selectedSourceSummary.total,
    selectedSources.length,
    t
  ])

  const executeCheckNow = useCallback(async (sourceIds: number[]) => {
    const normalizedIds = normalizeSourceIds(sourceIds)
    if (!isCurrentAccount() || normalizedIds.length === 0) return

    setCheckingSourceIds((prev) => Array.from(new Set([...prev, ...normalizedIds])))
    try {
      const result = await checkWatchlistSourcesNow(normalizedIds)
      const successCount = typeof result.success === "number"
        ? result.success
        : (Array.isArray(result.items) ? result.items.filter((item) => item.status === "ok").length : 0)
      const failedCount = typeof result.failed === "number"
        ? result.failed
        : Math.max(normalizedIds.length - successCount, 0)
      const viewRunsAction = (
        <Button
          type="link"
          size="small"
          className="px-0"
          onClick={() => setActiveTab("runs")}
        >
          {t("watchlists:sources.viewRuns", "View Runs")}
        </Button>
      )

      if (failedCount === 0) {
        message.success({
          content: (
            <span>
              {t(
                "watchlists:sources.checkNowSuccess",
                "Checked {{count}} source{{plural}}",
                {
                  count: successCount,
                  plural: successCount === 1 ? "" : "s"
                }
              )}{" "}
              {viewRunsAction}
            </span>
          )
        })
      } else if (successCount > 0) {
        message.warning({
          content: (
            <span>
              {t(
                "watchlists:sources.checkNowPartial",
                "Checked {{success}} source{{successPlural}}, {{failed}} failed",
                {
                  success: successCount,
                  successPlural: successCount === 1 ? "" : "s",
                  failed: failedCount
                }
              )}{" "}
              {viewRunsAction}
            </span>
          )
        })
      } else {
        message.error(t("watchlists:sources.checkNowError", "Failed to check selected sources"))
      }

      await invalidateWatchlists("sources", "source-seen", "job-preview", "runs", "items", "outputs", "overview")
    } catch (err) {
      console.error("Failed to check sources now:", err)
      message.error(t("watchlists:sources.checkNowError", "Failed to check selected sources"))
    } finally {
      setCheckingSourceIds((prev) => prev.filter((id) => !normalizedIds.includes(id)))
    }
  }, [invalidateWatchlists, isCurrentAccount, setActiveTab, t])

  const requestCheckNow = useCallback((sourceIds: number[]) => {
    const normalizedIds = normalizeSourceIds(sourceIds)
    if (normalizedIds.length === 0) return

    if (shouldConfirmMultiSourceCheck(normalizedIds)) {
      Modal.confirm({
        title: t(
          "watchlists:sources.checkNowMultiConfirmTitle",
          "Check {{count}} sources now?",
          { count: normalizedIds.length }
        ),
        content: t(
          "watchlists:sources.checkNowMultiConfirmDescription",
          "This will manually force a refresh for all selected sources."
        ),
        okText: t("watchlists:sources.checkNow", "Check Now"),
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => executeCheckNow(normalizedIds)
      })
      return
    }

    void executeCheckNow(normalizedIds)
  }, [executeCheckNow, t])

  const toggleSourceSelection = useCallback((source: WatchlistSource) => {
    setSelectedRowKeys((previousKeys) => {
      const selectedSet = new Set(previousKeys.map((key) => String(key)))
      const sourceKey = String(source.id)
      if (selectedSet.has(sourceKey)) {
        return previousKeys.filter((key) => String(key) !== sourceKey)
      }
      return [...previousKeys, source.id]
    })
  }, [])

  const handleOpenNewSourceForm = useCallback(() => {
    setSourceCloneDraft(null)
    openSourceForm()
  }, [openSourceForm])

  const handleOpenExistingSourceForm = useCallback((sourceId: number) => {
    setSourceCloneDraft(null)
    openSourceForm(sourceId)
  }, [openSourceForm])

  const handleCloseSourceForm = useCallback(() => {
    setSourceCloneDraft(null)
    closeSourceForm()
  }, [closeSourceForm])

  // Handle form submit
  const handleFormSubmit = async (
    values: {
      name: string
      url: string
      source_type: SourceType
      tags: string[]
      settings?: Record<string, unknown> | null
    }
  ) => {
    try {
      if (!isCurrentAccount()) return
      if (sourceFormEditId) {
        await updateWatchlistSource(sourceFormEditId, values)
        message.success(t("watchlists:sources.updated", "Source updated"))
      } else {
        await createWatchlistSource({
          ...values,
          ...(sourceCloneDraft
            ? {
                active: false,
                group_ids: sourceCloneDraft.group_ids,
                watchlist_id: sourceCloneDraft.watchlist_id ?? selectedWatchlistId ?? undefined
              }
            : {
                watchlist_id: selectedWatchlistId ?? undefined
              })
        })
        message.success(
          sourceCloneDraft
            ? t(
                "watchlists:sources.cloneSuccess",
                "Feed cloned as an inactive copy. Review and enable it when ready."
              )
            : t("watchlists:sources.created", "Source created")
        )
      }
      handleCloseSourceForm()
      void refreshSourceCatalog()
    } catch (err) {
      console.error("Failed to save source:", err)
      message.error(t("watchlists:sources.saveError", "Failed to save source"))
    }
  }

  const handleExport = async () => {
    try {
      const opml = await exportOpml({
        tag: selectedTagName ? [selectedTagName] : undefined,
        group: selectedGroupId ? [selectedGroupId] : undefined,
        type: selectedTypeFilter || undefined
      })
      const blob = new Blob([opml], { type: "application/xml" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `watchlists_sources_${Date.now()}.opml`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      message.success(t("watchlists:sources.exported", "OPML exported"))
    } catch (err) {
      console.error("Failed to export OPML:", err)
      message.error(t("watchlists:sources.exportError", "Failed to export OPML"))
    }
  }

  const handleCloneSource = (source: WatchlistSource) => {
    setSourceCloneDraft(buildClonedWatchlistSourcePayload(source, selectedWatchlistId))
    openSourceForm()
  }

  // Get source for editing
  const editingSource = sourceFormEditId
    ? sources.find((s) => s.id === sourceFormEditId)
    : undefined
  const sourceFormInitialValues = editingSource ?? sourceCloneDraft ?? undefined
  const sourceFormMode = sourceFormEditId ? "edit" : "create"

  // Table columns
  const tagsColumn: ColumnsType<WatchlistSource>[number] = {
    title: t("watchlists:sources.columns.tags", "Tags"),
    dataIndex: "tags",
    key: "tags",
    width: 200,
    render: (tags: string[]) => (
      <div className="flex flex-wrap gap-1">
        {tags.slice(0, 3).map((tag) => (
          <Tag key={tag} className="text-xs">
            {tag}
          </Tag>
        ))}
        {tags.length > 3 && (
          <Tag className="text-xs">+{tags.length - 3}</Tag>
        )}
      </div>
    )
  }

  const columns: ColumnsType<WatchlistSource> = [
    {
      title: t("watchlists:sources.columns.name", "Name"),
      dataIndex: "name",
      key: "name",
      width: 260,
      render: (name: string, record) => {
        const safeUrl = safeExternalUrl(record.url)
        return (
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <span className="font-medium">{name}</span>
            {safeUrl && (
              <Tooltip title={record.url}>
                <a
                  href={safeUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={t(
                    "watchlists:accessibilityHardening.source.openWebsite",
                    "Open source website: {{name}}",
                    { name: record.name }
                  )}
                  className="text-text-subtle hover:text-text-muted"
                  onClick={(e) => e.stopPropagation()}
                >
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden />
                </a>
              </Tooltip>
            )}
          </div>
          {!showAdvancedColumns && (
            <div className="text-xs text-text-muted" data-testid={`source-compact-summary-${record.id}`}>
              {t(
                "watchlists:sources.compactSummary",
                "{{groups}} {{groupLabel}} • {{tags}} {{tagLabel}}",
                {
                  groups: toNormalizedGroupIds(record).length,
                  groupLabel:
                    toNormalizedGroupIds(record).length === 1
                      ? t("watchlists:sources.compactSummaryGroupSingular", "group")
                      : t("watchlists:sources.compactSummaryGroupPlural", "groups"),
                  tags: (record.tags || []).length,
                  tagLabel:
                    (record.tags || []).length === 1
                      ? t("watchlists:sources.compactSummaryTagSingular", "tag")
                      : t("watchlists:sources.compactSummaryTagPlural", "tags")
                }
              )}
            </div>
          )}
        </div>
        )
      }
    },
    {
      title: t("watchlists:sources.columns.type", "Type"),
      dataIndex: "source_type",
      key: "source_type",
      width: 100,
      render: (type: SourceType) => (
        <Tag color={SOURCE_TYPE_COLORS[type] || "default"}>
          {type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: t("watchlists:sources.columns.status", "Status"),
      key: "status",
      width: 300,
      render: (_, record) => {
        const statusVisual = getSourceStatusVisual(record.status, record.active)
        const health = sourceHealthById?.[record.id]
        const deferUntil = health?.defer_until
        const consecutiveNotModified = health?.consec_not_modified ?? 0
        const hasBackoff = typeof deferUntil === "string" && deferUntil.length > 0

        return (
          <div className="flex flex-wrap items-center gap-1">
            <Tag color={statusVisual.color}>
              <span className="inline-flex items-center gap-1">
                {renderSourceStatusIcon(statusVisual.iconToken)}
                <span>{statusVisual.label}</span>
              </span>
            </Tag>
            {consecutiveNotModified > 0 && (
              <Tag color={consecutiveNotModified >= 5 ? "red" : "gold"}>
                {t("watchlists:sources.staleCount", "Stale x{{count}}", {
                  count: consecutiveNotModified
                })}
              </Tag>
            )}
            {hasBackoff && (
              <Tooltip title={new Date(deferUntil).toLocaleString()}>
                <Tag color="gold">
                  {t("watchlists:sources.backoffUntil", "Backoff {{time}}", {
                    time: formatRelativeTime(deferUntil, t, { compact: true })
                  })}
                </Tag>
              </Tooltip>
            )}
          </div>
        )
      }
    },
    {
      title: t("watchlists:sources.columns.lastScraped", "Last Scraped"),
      dataIndex: "last_scraped_at",
      key: "last_scraped_at",
      width: 190,
      render: (date: string | null, record) => {
        const checkNowLoading = checkingSourceIds.includes(record.id)
        const checkNowTooltip = t(
          "watchlists:accessibilityHardening.source.checkNow",
          "Check now: {{name}}",
          { name: record.name }
        )

        return (
          <div className="flex items-center gap-1.5">
            {date ? (
              <span className="text-sm text-text-muted">
                {formatRelativeTime(date, t)}
              </span>
            ) : (
              <span className="text-sm text-text-subtle">
                {t("watchlists:sources.never", "Never")}
              </span>
            )}
            <Tooltip title={checkNowTooltip}>
              <Button
                type="text"
                size="small"
                shape="circle"
                icon={<RefreshCw className="h-3.5 w-3.5" aria-hidden />}
                loading={checkNowLoading}
                aria-label={checkNowTooltip}
                onClick={(event) => {
                  event.stopPropagation()
                  requestCheckNow([record.id])
                }}
              />
            </Tooltip>
          </div>
        )
      }
    },
    {
      title: t("watchlists:sources.columns.active", "Active"),
      dataIndex: "active",
      key: "active",
      width: 140,
      align: "center",
      render: (active: boolean, record) => (
        <span className="inline-flex items-center gap-2">
          <Switch
            checked={active}
            size="small"
            aria-label={t("watchlists:accessibilityHardening.source.toggleActive", "Toggle active: {{name}}", { name: record.name })}
            onChange={() => handleToggleActive(record)}
          />
          <span className="text-xs text-text-muted">
            {active
              ? t("common:enabled", "Enabled")
              : t("common:disabled", "Disabled")}
          </span>
        </span>
      )
    },
    {
      title: t("watchlists:sources.columns.actions", "Actions"),
      key: "actions",
      width: 140,
      align: "center",
      render: (_, record) => (
        <Space size="small">
          <Tooltip title={t("common:edit", "Edit")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:accessibilityHardening.source.edit", "Edit source: {{name}}", { name: record.name })}
              icon={<Edit2 className="h-4 w-4" aria-hidden />}
              onClick={() => handleOpenExistingSourceForm(record.id)}
            />
          </Tooltip>
          <Tooltip title={t("watchlists:sources.seenInfo", "Source Health & Dedup Stats")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:accessibilityHardening.source.openHealth", "Open source health: {{name}}", { name: record.name })}
              icon={<Eye className="h-4 w-4" aria-hidden />}
              onClick={() => setSeenDrawerSourceId(record.id)}
            />
          </Tooltip>
          <Tooltip title={t("watchlists:sources.clone", "Clone")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:sources.cloneFeedAria", "Clone {{name}}", { name: record.name })}
              icon={<Copy className="h-4 w-4" />}
              onClick={() => handleCloneSource(record)}
            />
          </Tooltip>
          <Tooltip title={t("common:delete", "Delete")}>
            <Button
              type="text"
              size="small"
              danger
              aria-label={t("watchlists:accessibilityHardening.source.delete", "Delete source: {{name}}", { name: record.name })}
              icon={<Trash2 className="h-4 w-4" aria-hidden />}
              onClick={() => void requestDeleteConfirmation(record)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const tableColumns = showAdvancedColumns
    ? [columns[0], columns[1], columns[2], tagsColumn, columns[3], columns[4], columns[5]]
    : columns

  const renderSourceStatus = (source: WatchlistSource) => {
    const statusVisual = getSourceStatusVisual(source.status, source.active)
    const health = sourceHealthById?.[source.id]
    const deferUntil = health?.defer_until
    const consecutiveNotModified = health?.consec_not_modified ?? 0
    const hasBackoff = typeof deferUntil === "string" && deferUntil.length > 0

    return (
      <div className="flex flex-wrap items-center gap-1">
        <Tag color={statusVisual.color}>
          <span className="inline-flex items-center gap-1">
            {renderSourceStatusIcon(statusVisual.iconToken)}
            <span>{statusVisual.label}</span>
          </span>
        </Tag>
        {consecutiveNotModified > 0 && (
          <Tag color={consecutiveNotModified >= 5 ? "red" : "gold"}>
            {t("watchlists:sources.staleCount", "Stale x{{count}}", {
              count: consecutiveNotModified
            })}
          </Tag>
        )}
        {hasBackoff && (
          <Tag color="gold">
            {t("watchlists:sources.backoffUntil", "Backoff {{time}}", {
              time: formatRelativeTime(deferUntil, t, { compact: true })
            })}
          </Tag>
        )}
      </div>
    )
  }

  const renderConstrainedSourceList = () => (
    <div className="space-y-3" data-testid="watchlists-sources-constrained-list">
      {(Array.isArray(sources) ? sources : []).map((source) => {
        const groupCount = toNormalizedGroupIds(source).length
        const groupLabel = groupCount === 1
          ? t("watchlists:sources.compactSummaryGroupSingular", "group")
          : t("watchlists:sources.compactSummaryGroupPlural", "groups")
        const sourceTags = Array.isArray(source.tags) ? source.tags : []
        const checkNowLoading = checkingSourceIds.includes(source.id)
        const isSelected = selectedRowKeys.some((key) => String(key) === String(source.id))
        const sourceSafeUrl = safeExternalUrl(source.url)
        const sourceWebsiteLabel = t(
          "watchlists:accessibilityHardening.source.openWebsite",
          "Open source website: {{name}}",
          { name: source.name }
        )

        return (
          <article
            key={source.id}
            className="rounded-lg border border-border bg-surface p-3"
            data-testid={`watchlists-source-card-${source.id}`}
          >
            <div className="flex items-start justify-between gap-3">
              <label
                htmlFor={`watchlists-source-select-${source.id}`}
                className="flex min-w-0 items-start gap-3"
              >
                <input
                  id={`watchlists-source-select-${source.id}`}
                  type="checkbox"
                  className="mt-1 h-4 w-4 shrink-0"
                  checked={isSelected}
                  aria-label={t("watchlists:sources.selectSourceAria", "Select {{name}}", { name: source.name })}
                  onChange={() => toggleSourceSelection(source)}
                />
                <span className="min-w-0">
                  <span className="block truncate font-medium text-text">{source.name}</span>
                </span>
              </label>
              <Tag color={SOURCE_TYPE_COLORS[source.source_type] || "default"} className="shrink-0">
                {source.source_type.toUpperCase()}
              </Tag>
            </div>
            {sourceSafeUrl ? (
              <a
                href={sourceSafeUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={sourceWebsiteLabel}
                className="mt-1 inline-flex min-h-11 items-center gap-1 text-xs text-text-muted hover:text-primary"
                onClick={(event) => event.stopPropagation()}
              >
                {sourceWebsiteLabel}
                <ExternalLink className="h-3.5 w-3.5" aria-hidden />
              </a>
            ) : null}

            <div className="mt-3 flex flex-wrap gap-1.5">
              {sourceTags.slice(0, 4).map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
              {sourceTags.length > 4 ? <Tag>+{sourceTags.length - 4}</Tag> : null}
              <Tag>
                {t("watchlists:sources.groupCount", "{{count}} {{label}}", {
                  count: groupCount,
                  label: groupLabel
                })}
              </Tag>
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              {renderSourceStatus(source)}
              <span className="text-xs text-text-muted">
                {source.last_scraped_at
                  ? formatRelativeTime(source.last_scraped_at, t)
                  : t("watchlists:sources.never", "Never")}
              </span>
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <span className="inline-flex items-center gap-2">
                <Switch
                  checked={source.active}
                  size="small"
                  aria-label={t("watchlists:accessibilityHardening.source.toggleActive", "Toggle active: {{name}}", { name: source.name })}
                  onChange={() => handleToggleActive(source)}
                />
                <span className="text-xs text-text-muted">
                  {source.active ? t("common:enabled", "Enabled") : t("common:disabled", "Disabled")}
                </span>
              </span>
              <Space size="small">
                <Button
                  type="text"
                  size="small"
                  className="min-h-11 min-w-11"
                  aria-label={t("watchlists:accessibilityHardening.source.checkNow", "Check now: {{name}}", { name: source.name })}
                  icon={<RefreshCw className="h-4 w-4" aria-hidden />}
                  loading={checkNowLoading}
                  onClick={() => requestCheckNow([source.id])}
                />
                <Button
                  type="text"
                  size="small"
                  className="min-h-11 min-w-11"
                  aria-label={t("watchlists:accessibilityHardening.source.openHealth", "Open source health: {{name}}", { name: source.name })}
                  icon={<Eye className="h-4 w-4" aria-hidden />}
                  onClick={() => setSeenDrawerSourceId(source.id)}
                />
                <Button
                  type="text"
                  size="small"
                  className="min-h-11 min-w-11"
                  aria-label={t("watchlists:sources.cloneFeedAria", "Clone {{name}}", { name: source.name })}
                  icon={<Copy className="h-4 w-4" />}
                  onClick={() => handleCloneSource(source)}
                />
                <Button
                  type="text"
                  size="small"
                  className="min-h-11 min-w-11"
                  aria-label={t("watchlists:accessibilityHardening.source.edit", "Edit source: {{name}}", { name: source.name })}
                  icon={<Edit2 className="h-4 w-4" aria-hidden />}
                  onClick={() => handleOpenExistingSourceForm(source.id)}
                />
                <Button
                  type="text"
                  size="small"
                  className="min-h-11 min-w-11"
                  danger
                  aria-label={t("watchlists:accessibilityHardening.source.delete", "Delete source: {{name}}", { name: source.name })}
                  icon={<Trash2 className="h-4 w-4" aria-hidden />}
                  onClick={() => void requestDeleteConfirmation(source)}
                />
              </Space>
            </div>
          </article>
        )
      })}
      <div className="text-xs text-text-subtle">
        {t("watchlists:sources.totalItems", "{{total}} sources", { total: sourcesTotal })}
      </div>
      {sourcesTotal > sourcesPageSize && (
        <Pagination
          current={sourcesPage}
          pageSize={sourcesPageSize}
          total={sourcesTotal}
          showSizeChanger
          showTotal={(total) => t("watchlists:sources.totalItems", "{{total}} sources", { total })}
          onChange={(page, pageSize) => {
            setSourcesPage(page)
            if (pageSize !== sourcesPageSize) {
              setSourcesPageSize(pageSize)
            }
          }}
        />
      )}
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <Search
            placeholder={t("watchlists:sources.searchPlaceholder", "Search sources...")}
            value={sourcesSearch}
            onChange={(e) => setSourcesSearch(e.target.value)}
            onSearch={refreshSources}
            allowClear
            className="w-64"
          />
          <Select
            placeholder={t("watchlists:sources.filterByTag", "Filter by tag")}
            value={selectedTagName}
            onChange={setSelectedTagName}
            allowClear
            className="w-40"
            options={(Array.isArray(tags) ? tags : []).map((tag) => ({
              label: tag.name,
              value: tag.name
            }))}
          />
          <Select
            placeholder={t("watchlists:sources.filterByType", "Filter by type")}
            value={selectedTypeFilter}
            onChange={(value) => {
              setSelectedTypeFilter(value)
              setSourcesPage(1)
            }}
            allowClear
            className="w-32"
            options={[
              { label: "RSS", value: "rss" },
              { label: "Site", value: "site" },
              { label: "Forum", value: "forum" }
            ]}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type={showAdvancedColumns ? "default" : "dashed"}
            data-testid="watchlists-sources-advanced-toggle"
            onClick={() => setShowAdvancedColumns((previous) => !previous)}
          >
            {showAdvancedColumns
              ? t("watchlists:sources.hideAdvancedDetails", "Hide advanced details")
              : t("watchlists:sources.showAdvancedDetails", "Show advanced details")}
          </Button>
          <Button
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={refreshSources}
            loading={sourcesLoading}
          >
            {t("common:refresh", "Refresh")}
          </Button>
          <Button
            icon={<Download className="h-4 w-4" />}
            onClick={handleExport}
          >
            {t("watchlists:sources.export", "Export OPML")}
          </Button>
          <Tooltip title={t("watchlists:sources.importTooltip", "Import feeds from an OPML file (standard RSS reader export format)")}>
            <Button
              icon={<UploadCloud className="h-4 w-4" />}
              onClick={() => setImportOpen(true)}
            >
              {t("watchlists:sources.import", "Import OPML")}
            </Button>
          </Tooltip>
          <Button
            type="primary"
            icon={<Plus className="h-4 w-4" />}
            onClick={handleOpenNewSourceForm}
          >
            {t("watchlists:sources.addSource", "Add Source")}
          </Button>
        </div>
      </div>

      {!showAdvancedColumns && (
        <div className="text-xs text-text-muted" data-testid="watchlists-sources-density-hint">
          {t(
            "watchlists:sources.metricsHint",
            "Showing core columns. Use advanced mode for tag-level source details."
          )}
        </div>
      )}

      {sourcesLoadError && (
        <Alert
          type={sourcesLoadError.severity}
          showIcon
          title={sourcesLoadError.title}
          description={sourcesLoadError.description}
          action={(
            <Button
              size="small"
              onClick={() => void sourcesQuery.refetch()}
              loading={sourcesLoading}
            >
              {t("watchlists:errors.retry", "Retry")}
            </Button>
          )}
        />
      )}

      {selectedRowKeys.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-dashed border-border p-3 text-sm">
          <span className="text-text-muted">
            {t("watchlists:sources.selectedCount", "{{count}} selected", { count: selectedRowKeys.length })}
          </span>
          <Button
            size="small"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            aria-label={t(
              "watchlists:sources.checkNowSelectedTooltip",
              "Check now for {{count}} selected sources",
              { count: selectedSourceIds.length }
            )}
            onClick={() => requestCheckNow(selectedSourceIds)}
            loading={selectedSourceIds.some((id) => checkingSourceIds.includes(id))}
            disabled={bulkWorking}
          >
            {t("watchlists:sources.checkNow", "Check Now")}
          </Button>
          <Button size="small" onClick={() => requestBulkToggle(true)} loading={bulkWorking}>
            {t("watchlists:sources.bulkEnable", "Enable")}
          </Button>
          <Button size="small" onClick={() => requestBulkToggle(false)} loading={bulkWorking}>
            {t("watchlists:sources.bulkDisable", "Disable")}
          </Button>
          <Select
            size="small"
            allowClear
            className="w-52"
            placeholder={t("watchlists:sources.moveSelectGroup", "Move to group")}
            value={bulkMoveTargetValue}
            onChange={(value) => setBulkMoveTargetValue((value as BulkMoveTargetValue) ?? null)}
            options={[
              {
                value: BULK_MOVE_NO_GROUP_VALUE,
                label: t("watchlists:sources.moveTargetNone", "No group")
              },
              ...(Array.isArray(groups) ? groups : []).map((group) => ({
                value: group.id,
                label: group.name
              }))
            ]}
            disabled={bulkWorking}
          />
          <Button
            size="small"
            onClick={requestBulkMoveToGroup}
            loading={bulkWorking}
            disabled={bulkMoveTargetValue == null}
          >
            {t("watchlists:sources.bulkMove", "Move")}
          </Button>
          <Button size="small" danger loading={bulkWorking} onClick={() => void requestBulkDelete()}>
            {t("watchlists:sources.bulkDelete", "Delete")}
          </Button>
          <Button size="small" onClick={() => setSelectedRowKeys([])}>
            {t("common:clear", "Clear")}
          </Button>
        </div>
      )}

      {unifiedEmptyState ? (
        <div className="rounded-lg border border-dashed border-border bg-surface p-8">
          <Empty
            description={
              <div className="space-y-2">
                <p className="text-text-muted">
                  {t("watchlists:sources.emptyInitialTitle", "No watchlist sources yet")}
                </p>
                <p className="text-sm text-text-subtle">
                  {t(
                    "watchlists:sources.emptyInitialDescription",
                    "Add a source or import OPML to start monitoring updates."
                  )}
                </p>
              </div>
            }
          >
            <Space orientation="vertical" align="center" size="middle">
              <Button
                type="primary"
                size="large"
                onClick={() => setActiveTab("overview")}
              >
                {t("watchlists:sources.quickSetupCta", "Get started with Quick Setup")}
              </Button>
              <Space>
                <Button
                  icon={<Plus className="h-4 w-4" />}
                  onClick={handleOpenNewSourceForm}
                >
                  {t("watchlists:sources.addSource", "Add Source")}
                </Button>
                <Tooltip title={t("watchlists:sources.importTooltip", "Import feeds from an OPML file (standard RSS reader export format)")}>
                  <Button
                    icon={<UploadCloud className="h-4 w-4" />}
                    onClick={() => setImportOpen(true)}
                  >
                    {t("watchlists:sources.import", "Import OPML")}
                  </Button>
                </Tooltip>
              </Space>
            </Space>
          </Empty>
        </div>
      ) : (
        <div className="flex flex-col gap-4 lg:flex-row">
          <div className="w-full lg:w-72 shrink-0 space-y-4">
            <GroupsTree
              groups={groups}
              selectedGroupId={selectedGroupId}
              loading={groupsLoading}
              onSelect={setSelectedGroupId}
              onRefresh={() => void refreshSourceCatalog()}
            />
          </div>

          <div className="flex-1">
            {isConstrained ? (
              renderConstrainedSourceList()
            ) : (
              <Table
                dataSource={Array.isArray(sources) ? sources : []}
                columns={tableColumns}
                rowKey="id"
                aria-label={t("watchlists:sources.tableAria", "Feeds table")}
                loading={sourcesLoading}
                locale={{
                  emptyText: (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description={t(
                        "watchlists:sources.emptyTableDescription",
                        tableEmptyDescription
                      )}
                    />
                  )
                }}
                rowSelection={{
                  selectedRowKeys,
                  onChange: setSelectedRowKeys
                }}
                pagination={{
                  current: sourcesPage,
                  pageSize: sourcesPageSize,
                  total: sourcesTotal,
                  showSizeChanger: true,
                  showTotal: (total) =>
                    t("watchlists:sources.totalItems", "{{total}} sources", { total }),
                  onChange: (page, pageSize) => {
                    setSourcesPage(page)
                    if (pageSize !== sourcesPageSize) {
                      setSourcesPageSize(pageSize)
                    }
                  }
                }}
                size="middle"
                scroll={{ x: "max-content" }}
              />
            )}
          </div>
        </div>
      )}

      <SourceFormModal
        open={sourceFormOpen}
        onClose={handleCloseSourceForm}
        onSubmit={handleFormSubmit}
        initialValues={sourceFormInitialValues}
        mode={sourceFormMode}
        existingTags={(Array.isArray(tags) ? tags : []).map((t) => t.name)}
        forumsEnabled={Boolean(settings?.forums_enabled)}
      />

      <SourcesBulkImport
        open={importOpen}
        onClose={() => setImportOpen(false)}
        groups={groups}
        tags={tags}
        defaultGroupId={selectedGroupId}
        watchlistId={selectedWatchlistId}
        onImported={() => void refreshSourceCatalog()}
      />

      <SourceSeenDrawer
        open={seenDrawerSourceId !== null}
        onClose={() => setSeenDrawerSourceId(null)}
        sourceId={seenDrawerSourceId}
        sourceName={sources.find((s) => s.id === seenDrawerSourceId)?.name}
      />
    </div>
  )
}
