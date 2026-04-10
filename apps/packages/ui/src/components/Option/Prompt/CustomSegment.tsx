import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Alert,
  Input,
  Pagination,
  Segmented,
  Select,
  Skeleton,
  Tooltip,
  notification,
  type InputRef
} from "antd"
import {
  ChevronDown,
  ChevronUp,
  Cloud,
  Download,
  Filter,
  FolderPlus,
  Keyboard,
  LayoutGrid,
  List,
  Star,
  StarOff,
  Trash2,
  UploadCloud
} from "lucide-react"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { useDebounce } from "@/hooks/useDebounce"
import { PromptActionsMenu } from "./PromptActionsMenu"
import { SyncStatusBadge } from "./SyncStatusBadge"
import { PromptBulkActionBar } from "./PromptBulkActionBar"
import {
  PromptListTable,
  type PromptTableDensity
} from "./PromptListTable"
import { PromptListToolbar } from "./PromptListToolbar"
import type { PromptGalleryDensity } from "./PromptGalleryCard"
import type { PromptListQueryState, PromptRowVM } from "./prompt-workspace-types"
import type { TagMatchMode } from "./custom-prompts-utils"
import { useContextualHints } from "./useContextualHints"
import { useFilterPresets, type FilterPreset } from "./useFilterPresets"
import { usePromptWorkspace } from "./PromptWorkspaceProvider"
import { usePromptSync } from "./hooks/usePromptSync"
import { usePromptEditor } from "./hooks/usePromptEditor"
import { usePromptBulkActions } from "./hooks/usePromptBulkActions"
import { usePromptImportExport } from "./hooks/usePromptImportExport"
import { usePromptCollections } from "./hooks/usePromptCollections"
import { usePromptFilteredData } from "./hooks/usePromptFilteredData"

const PromptGalleryCard = React.lazy(() =>
  import("./PromptGalleryCard").then((module) => ({
    default: module.PromptGalleryCard
  }))
)
const PromptStarterCards = React.lazy(() =>
  import("./PromptStarterCards").then((module) => ({
    default: module.PromptStarterCards
  }))
)
const ContextualHint = React.lazy(() =>
  import("./ContextualHint").then((module) => ({ default: module.ContextualHint }))
)

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

type PromptSortKey = "title" | "modifiedAt" | null
type PromptSortOrder = "ascend" | "descend" | null
type PromptSortState = { key: PromptSortKey; order: PromptSortOrder }
type PromptViewMode = "table" | "gallery"
type PromptSavedView = "all" | "favorites" | "recent" | "most_used" | "untagged"

const SORT_KEY = "tldw-prompts-custom-sort-v1"
const TABLE_DENSITY_KEY = "tldw-prompts-table-density-v1"
const VIEW_MODE_KEY = "tldw-prompts-view-mode-v1"
const GALLERY_DENSITY_KEY = "tldw-prompts-gallery-density-v1"

const readSort = (): PromptSortState => {
  if (typeof window === "undefined") return { key: null, order: null }
  try {
    const raw = window.sessionStorage.getItem(SORT_KEY)
    if (!raw) return { key: null, order: null }
    const p = JSON.parse(raw) as PromptSortState
    const keys: PromptSortKey[] = ["title", "modifiedAt", null]
    const orders: PromptSortOrder[] = ["ascend", "descend", null]
    if (!keys.includes(p?.key) || !orders.includes(p?.order)) return { key: null, order: null }
    return p
  } catch { return { key: null, order: null } }
}
const readTableDensity = (): PromptTableDensity => {
  if (typeof window === "undefined") return "comfortable"
  try { const r = window.localStorage.getItem(TABLE_DENSITY_KEY); return r === "compact" || r === "dense" || r === "comfortable" ? r : "comfortable" } catch { return "comfortable" }
}
const readViewMode = (): PromptViewMode => {
  if (typeof window === "undefined") return "table"
  try { const r = window.localStorage.getItem(VIEW_MODE_KEY); return r === "table" || r === "gallery" ? r : "table" } catch { return "table" }
}
const readGalleryDensity = (): PromptGalleryDensity => {
  if (typeof window === "undefined") return "rich"
  try { const r = window.localStorage.getItem(GALLERY_DENSITY_KEY); return r === "rich" || r === "compact" ? r : "rich" } catch { return "rich" }
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CustomSegmentProps {
  /** Project filter from URL (?project=) */
  projectFilter: string | null
  clearProjectFilter: () => void
  /** Callbacks to open shared modals */
  onOpenShortcutsHelp: () => void
  onQuickTest: (prompt: any) => void
  onUsePromptInChat: (prompt: any) => Promise<void>
  onOpenInspector: (promptId: string) => void
  closeInspector: () => void
  /** Open the shared bulk keyword modal (state lives in orchestrator) */
  onOpenBulkKeywordModal: (selectedIds: string[]) => void
  /** Ref updated by parent after bulk keyword operation to sync selection */
  bulkSelectionSyncRef?: React.RefObject<((ids: React.Key[]) => void) | null>
  /** Search input ref for keyboard shortcut "/" */
  searchInputRef: React.RefObject<InputRef | null>
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CustomSegment({
  projectFilter,
  clearProjectFilter,
  onOpenShortcutsHelp,
  onQuickTest,
  onUsePromptInChat,
  onOpenInspector,
  closeInspector,
  onOpenBulkKeywordModal,
  bulkSelectionSyncRef,
  searchInputRef,
}: CustomSegmentProps) {
  const {
    queryClient,
    isOnline,
    t,
    isCompactViewport,
    selectedSegment,
    data,
    dataStatus: status,
    utils
  } = usePromptWorkspace()

  const {
    confirmDanger,
    guardPrivateMode,
    getPromptKeywords,
    getPromptTexts,
    getPromptType,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    formatRelativePromptTime,
    getPromptRecordById,
    isFireFoxPrivateMode
  } = utils

  // ---- Local filter/display state ----
  const [searchText, setSearchText] = useState("")
  const [typeFilter, setTypeFilter] = useState<"all" | "system" | "quick">("all")
  const [usageFilter, setUsageFilter] = useState<"all" | "used" | "unused">("all")
  const [tagFilter, setTagFilter] = useState<string[]>([])
  const [tagMatchMode, setTagMatchMode] = useState<TagMatchMode>("any")
  const [syncFilter, setSyncFilter] = useState<string>("all")
  const [currentPage, setCurrentPage] = useState(1)
  const [resultsPerPage, setResultsPerPage] = useState(20)
  const [tableDensity, setTableDensity] = useState<PromptTableDensity>(readTableDensity)
  const [viewMode, setViewMode] = useState<PromptViewMode>(readViewMode)
  const [galleryDensity, setGalleryDensity] = useState<PromptGalleryDensity>(readGalleryDensity)
  const [promptSort, setPromptSort] = useState<PromptSortState>(readSort)
  const [savedView, setSavedView] = useState<PromptSavedView>("all")
  const [mobileFiltersExpanded, setMobileFiltersExpanded] = useState(false)

  const { presets: filterPresets, savePreset: saveFilterPreset, deletePreset: deleteFilterPreset } = useFilterPresets()
  const { shouldShow: shouldShowHint, dismiss: dismissHint, markShown: markHintShown } = useContextualHints()

  const debouncedSearchText = useDebounce(searchText, 300)
  const normalizedSearchText = debouncedSearchText.trim()
  const shouldUseServerSearch = isOnline && normalizedSearchText.length > 0

  // ---- Segment-local hooks (DAG: sync -> editor -> bulk -> importExport -> collections) ----
  const sync = usePromptSync({ queryClient, isOnline, t })

  const editor = usePromptEditor({
    queryClient,
    isOnline,
    t,
    guardPrivateMode,
    getPromptTexts,
    getPromptKeywords,
    getPromptRecordById,
    confirmDanger,
    syncPromptAfterLocalSave: sync.syncPromptAfterLocalSave,
    onEmptyTrashSuccess: () => { bulk.setTrashSelectedRowKeys([]) }
  })

  const bulk = usePromptBulkActions({
    queryClient,
    data,
    isOnline,
    isFireFoxPrivateMode,
    t,
    guardPrivateMode,
    getPromptKeywords,
    buildPromptUpdatePayload: editor.buildPromptUpdatePayload,
    confirmDanger
  })

  // Expose bulk selection setter so orchestrator can sync selection after shared bulk operations
  useEffect(() => {
    if (bulkSelectionSyncRef) {
      (bulkSelectionSyncRef as React.MutableRefObject<((ids: React.Key[]) => void) | null>).current = bulk.setSelectedRowKeys
    }
    return () => {
      if (bulkSelectionSyncRef) {
        (bulkSelectionSyncRef as React.MutableRefObject<((ids: React.Key[]) => void) | null>).current = null
      }
    }
  }, [bulkSelectionSyncRef, bulk.setSelectedRowKeys])

  const importExport = usePromptImportExport({
    queryClient,
    data,
    isOnline,
    t,
    guardPrivateMode,
    confirmDanger
  })

  const collections = usePromptCollections({
    queryClient,
    isOnline,
    t,
    setSelectedRowKeys: bulk.setSelectedRowKeys
  })

  // ---- Filtered data ----
  const filteredDataHook = usePromptFilteredData({
    data,
    isOnline,
    normalizedSearchText,
    shouldUseServerSearch,
    projectFilter,
    typeFilter,
    syncFilter,
    usageFilter,
    tagFilter,
    tagMatchMode,
    savedView,
    selectedCollection: collections.selectedCollection,
    currentPage,
    resultsPerPage,
    promptSort,
    getPromptKeywords,
    getPromptTexts,
    getPromptType,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    t
  })
  const {
    serverSearchStatus,
    allTags,
    pendingSyncCount,
    localSyncBatchPlan,
    sidebarCounts,
    sortedFilteredData,
    customPromptRows,
    tableTotal,
    hiddenServerResultsOnPage,
    useServerSearchResults
  } = filteredDataHook

  // ---- Effects: persist UI settings ----
  useEffect(() => { setCurrentPage(1) }, [normalizedSearchText, projectFilter, typeFilter, collections.collectionFilter, tagFilter, tagMatchMode])

  useEffect(() => { try { window.sessionStorage.setItem(SORT_KEY, JSON.stringify(promptSort)) } catch {} }, [promptSort])
  useEffect(() => { try { window.localStorage.setItem(TABLE_DENSITY_KEY, tableDensity) } catch {} }, [tableDensity])
  useEffect(() => { try { window.localStorage.setItem(VIEW_MODE_KEY, viewMode) } catch {} }, [viewMode])
  useEffect(() => { try { window.localStorage.setItem(GALLERY_DENSITY_KEY, galleryDensity) } catch {} }, [galleryDensity])

  useEffect(() => {
    if (!shouldUseServerSearch || serverSearchStatus !== "error") return
    notification.warning({
      message: t("managePrompts.searchServerFallback", { defaultValue: "Server search unavailable" }),
      description: t("managePrompts.searchServerFallbackDesc", { defaultValue: "Falling back to local search results for this query." })
    })
  }, [serverSearchStatus, shouldUseServerSearch, t])

  // Clear selection for items no longer visible
  useEffect(() => {
    const visibleIds = new Set(sortedFilteredData.map((p: any) => p.id))
    bulk.setSelectedRowKeys((prev) => {
      const stillVisible = prev.filter((key) => visibleIds.has(key as string))
      if (stillVisible.length !== prev.length) {
        if (stillVisible.length < prev.length && prev.length > 0) {
          notification.info({ message: t("managePrompts.selectionFiltered", { defaultValue: "Some selected items were filtered out" }), duration: 2 })
        }
        return stillVisible
      }
      return prev
    })
  }, [sortedFilteredData, t, bulk])

  // ---- Callbacks ----
  const handleLoadFilterPreset = useCallback((preset: FilterPreset) => {
    setTypeFilter(preset.typeFilter as any)
    setSyncFilter(preset.syncFilter as any)
    setTagFilter(preset.tagFilter)
    setTagMatchMode(preset.tagMatchMode)
    setSavedView(preset.savedView)
  }, [])

  const handleSaveFilterPreset = useCallback(
    (name: string) => {
      saveFilterPreset(name, { typeFilter, syncFilter, tagFilter, tagMatchMode, savedView })
    },
    [typeFilter, syncFilter, tagFilter, tagMatchMode, savedView, saveFilterPreset]
  )

  const handleCustomPromptTableQueryChange = useCallback(
    (patch: Partial<PromptListQueryState>) => {
      const nextPage = typeof patch.page === "number" ? patch.page : currentPage
      const nextPageSize = typeof patch.pageSize === "number" ? patch.pageSize : resultsPerPage
      if (nextPageSize !== resultsPerPage) { setResultsPerPage(nextPageSize); setCurrentPage(1) }
      else if (nextPage !== currentPage) { setCurrentPage(nextPage) }
      if (patch.sort) {
        const rawNextKey = patch.sort.key
        const nextKey: PromptSortKey = rawNextKey === "title" || rawNextKey === "modifiedAt" ? rawNextKey : null
        const nextOrder = patch.sort.order || null
        setPromptSort({ key: nextOrder ? nextKey : null, order: nextOrder })
      }
    },
    [currentPage, resultsPerPage]
  )

  // ---- Render callbacks (memoised) ----
  const renderCustomPromptTitleMeta = useCallback(
    (row: PromptRowVM) => {
      if (!isCompactViewport) return null
      return (
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="text-[11px] text-text-muted">
            {formatRelativePromptTime(row.updatedAt)}
          </span>
          <Tooltip
            title={!isOnline ? t("managePrompts.sync.offlineTooltip", { defaultValue: "Sync unavailable (offline). Showing last known status." }) : undefined}
          >
            <span className={!isOnline ? "opacity-60" : undefined}>
              <SyncStatusBadge
                syncStatus={row.syncStatus}
                sourceSystem={row.sourceSystem}
                serverId={row.serverId}
                compact
                onClick={
                  isOnline && row.syncStatus === "conflict"
                    ? () => sync.openConflictResolution(row.id)
                    : undefined
                }
              />
            </span>
          </Tooltip>
        </div>
      )
    },
    [formatRelativePromptTime, isCompactViewport, isOnline, sync.openConflictResolution, t]
  )

  const renderCustomPromptActions = useCallback(
    (row: PromptRowVM) => {
      const promptRecord = getPromptRecordById(row.id)
      const actionDisabled = isFireFoxPrivateMode || !promptRecord
      return (
        <PromptActionsMenu
          promptId={row.id}
          disabled={actionDisabled}
          syncStatus={row.syncStatus}
          serverId={row.serverId}
          inlineUseInChat={false}
          onEdit={() => { if (promptRecord) editor.openFullEditor(promptRecord) }}
          onDuplicate={() => { if (promptRecord) editor.handleDuplicatePrompt(promptRecord) }}
          onUseInChat={() => { if (promptRecord) void onUsePromptInChat(promptRecord) }}
          onQuickTest={() => { if (promptRecord) void onQuickTest(promptRecord) }}
          onDelete={() => { if (promptRecord) void editor.handleDeletePrompt(promptRecord) }}
          onShareLink={
            row.serverId && promptRecord
              ? () => { /* TODO: wire copyPromptShareLink */ }
              : undefined
          }
          onRetrySync={
            isOnline && promptRecord && row.syncStatus === "pending"
              ? () => { void sync.syncPromptAfterLocalSave(promptRecord.id) }
              : undefined
          }
          onPushToServer={
            isOnline && promptRecord
              ? () => { sync.setPromptToSync(promptRecord.id); sync.setProjectSelectorOpen(true) }
              : undefined
          }
          onPullFromServer={
            isOnline && row.serverId && promptRecord
              ? () => { sync.pullFromStudioMutation({ serverId: row.serverId as number, localId: promptRecord.id }) }
              : undefined
          }
          onUnlink={
            isOnline && row.serverId && promptRecord
              ? () => { sync.unlinkPromptMutation(promptRecord.id) }
              : undefined
          }
          onResolveConflict={
            isOnline && row.syncStatus === "conflict"
              ? () => { sync.openConflictResolution(row.id) }
              : undefined
          }
        />
      )
    },
    [getPromptRecordById, editor, onUsePromptInChat, onQuickTest, isFireFoxPrivateMode, isOnline, sync]
  )

  const customPromptsLoading = status === "pending" || (shouldUseServerSearch && serverSearchStatus === "pending")

  const customPromptTableQuery: PromptListQueryState = {
    searchText,
    typeFilter,
    syncFilter: syncFilter as PromptListQueryState["syncFilter"],
    usageFilter,
    tagFilter,
    tagMatchMode,
    sort: { key: promptSort.key, order: promptSort.order },
    page: currentPage,
    pageSize: resultsPerPage,
    savedView
  }

  const bulkActionTouchClass = isCompactViewport ? "min-h-[44px] px-3 py-2" : "px-2 py-1"

  // ---- JSX ----
  return (
    <div data-testid="prompts-custom">
      {/* Project filter banner */}
      {projectFilter && (
        <Alert
          type="info"
          showIcon
          className="mb-4"
          title={t("managePrompts.projectFilter.active", { defaultValue: "Filtering by project" })}
          description={t("managePrompts.projectFilter.description", {
            defaultValue: "Showing prompts linked to Project #{{projectId}}. Clear the filter to see all prompts.",
            projectId: projectFilter
          })}
          action={
            <button onClick={clearProjectFilter} className="text-sm text-primary hover:underline" data-testid="prompts-clear-project-filter">
              {t("managePrompts.projectFilter.clear", { defaultValue: "Show all prompts" })}
            </button>
          }
        />
      )}
      <div className="mb-6 space-y-3">
        {/* Bulk action bar */}
        {viewMode === "table" && bulk.selectedRowKeys.length > 0 && (
          <PromptBulkActionBar mode="legacy">
            <span className="text-sm text-primary">
              {t("managePrompts.bulk.selected", { defaultValue: "{{count}} selected", count: bulk.selectedRowKeys.length })}
            </span>
            <button onClick={() => bulk.triggerBulkExport()} data-testid="prompts-bulk-export"
              className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 ${bulkActionTouchClass}`}>
              <Download className="size-3" /> {t("managePrompts.bulk.export", { defaultValue: "Export selected" })}
            </button>
            <button onClick={() => onOpenBulkKeywordModal(bulk.selectedRowKeys.map((key) => String(key)))} disabled={bulk.isBulkAddingKeyword} data-testid="prompts-bulk-add-keyword"
              className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
              {t("managePrompts.bulk.addKeyword", { defaultValue: "Add keyword" })}
            </button>
            <button
              onClick={() => bulk.bulkToggleFavorite({ ids: bulk.selectedRowKeys.map((key) => String(key)), favorite: !bulk.allSelectedAreFavorite })}
              disabled={bulk.isBulkFavoriting} data-testid="prompts-bulk-toggle-favorite"
              className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
              {bulk.allSelectedAreFavorite ? <StarOff className="size-3" /> : <Star className="size-3" />}
              {bulk.allSelectedAreFavorite
                ? t("managePrompts.bulk.unfavorite", { defaultValue: "Unfavorite selected" })
                : t("managePrompts.bulk.favorite", { defaultValue: "Favorite selected" })}
            </button>
            {isOnline && (
              <button onClick={() => bulk.bulkPushToServer(bulk.selectedRowKeys.map((key) => String(key)))}
                disabled={bulk.isBulkPushing} data-testid="prompts-bulk-push-server"
                className={`inline-flex items-center gap-1 rounded border border-primary/30 text-sm text-primary hover:bg-primary/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
                <Cloud className="size-3" />
                {t("managePrompts.bulk.pushToServer", { defaultValue: "Push to server" })}
              </button>
            )}
            <button
              onClick={async () => {
                if (guardPrivateMode()) return
                const ok = await confirmDanger({
                  title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
                  content: t("managePrompts.bulk.deleteConfirm", { defaultValue: "Are you sure you want to delete {{count}} prompts?", count: bulk.selectedRowKeys.length }),
                  okText: t("common:delete", { defaultValue: "Delete" }),
                  cancelText: t("common:cancel", { defaultValue: "Cancel" })
                })
                if (!ok) return
                bulk.bulkDeletePrompts(bulk.selectedRowKeys as string[])
              }}
              disabled={bulk.isBulkDeleting} data-testid="prompts-bulk-delete"
              className={`inline-flex items-center gap-1 rounded border border-danger/30 text-sm text-danger hover:bg-danger/10 disabled:opacity-50 ${bulkActionTouchClass}`}>
              <Trash2 className="size-3" /> {t("managePrompts.bulk.delete", { defaultValue: "Delete selected" })}
            </button>
            <button onClick={() => bulk.setSelectedRowKeys([])} data-testid="prompts-clear-selection"
              className={`ml-auto inline-flex items-center rounded text-sm text-text-muted hover:text-text ${isCompactViewport ? "min-h-[44px] px-2" : ""}`}>
              {t("common:clearSelection", { defaultValue: "Clear selection" })}
            </button>
          </PromptBulkActionBar>
        )}

        {/* Batch sync status */}
        {isOnline && (sync.batchSyncState.running || sync.batchSyncState.failed.length > 0) && (
          <div data-testid="prompts-batch-sync-status" className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface2 p-2">
            {sync.batchSyncState.running ? (
              <span className="text-sm text-text-muted">
                {t("managePrompts.sync.batchProgress", { defaultValue: "Syncing {{completed}} of {{total}} prompts...", completed: sync.batchSyncState.completed, total: sync.batchSyncState.total })}
              </span>
            ) : (
              <span className="text-sm text-warn">
                {t("managePrompts.sync.batchFailedCount", { defaultValue: "{{count}} prompt(s) failed in the last batch run. Retry to continue.", count: sync.batchSyncState.failed.length })}
              </span>
            )}
          </div>
        )}

        {/* Collections panel */}
        {isOnline && (
          <div data-testid="prompts-collections-panel" className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface2 p-2">
            <Select
              value={collections.collectionFilter}
              onChange={(value) => collections.setCollectionFilter(value === "all" ? "all" : Number(value))}
              loading={collections.promptCollectionsStatus === "pending"}
              style={{ minWidth: isCompactViewport ? "100%" : 260 }}
              data-testid="prompts-collection-filter"
              options={[
                { label: t("managePrompts.collections.filterAll", { defaultValue: "All collections" }), value: "all" },
                ...collections.promptCollections.map((collection) => ({
                  label: `${collection.name} (${collection.prompt_ids?.length || 0})`,
                  value: collection.collection_id
                }))
              ]}
            />
            <button type="button" onClick={() => collections.setCreateCollectionModalOpen(true)} data-testid="prompts-collection-create"
              className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-2 text-sm font-medium text-text hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2">
              <FolderPlus className="size-4" />
              {t("managePrompts.collections.create", { defaultValue: "New collection" })}
            </button>
            {collections.selectedCollection && bulk.selectedRowKeys.length > 0 && (
              <button type="button"
                onClick={() => collections.addPromptsToCollectionMutation({ collection: collections.selectedCollection!, prompts: bulk.selectedPromptRows })}
                disabled={collections.isAssigningPromptCollection} data-testid="prompts-collection-add-selected"
                className="inline-flex items-center gap-2 rounded-md border border-primary/40 px-2 py-2 text-sm font-medium text-primary hover:bg-primary/10 disabled:opacity-50">
                {t("managePrompts.collections.addSelected", { defaultValue: "Add selected to collection" })}
              </button>
            )}
          </div>
        )}

        {/* Toolbar */}
        <PromptListToolbar mode="legacy" className="flex flex-wrap items-start justify-between gap-3 sm:items-center">
          {/* Left: Action buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <Tooltip title={t("managePrompts.newPromptHint", { defaultValue: "New prompt (N)" })}>
              <button onClick={() => editor.openFullEditor()} data-testid="prompts-add"
                className="inline-flex items-center rounded-md border border-transparent bg-primary px-2 py-2 text-md font-medium leading-4 text-white shadow-sm hover:bg-primaryStrong focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 disabled:opacity-50">
                {t("managePrompts.newPromptBtn", { defaultValue: "New prompt" })}
              </button>
            </Tooltip>
            <div className="inline-flex items-center rounded-md border border-border overflow-hidden">
              <button onClick={() => importExport.triggerExport()} data-testid="prompts-export"
                aria-label={t("managePrompts.exportLabel", { defaultValue: "Export prompts" })}
                className="inline-flex items-center gap-2 px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2">
                <Download className="size-4" /> {t("managePrompts.export", { defaultValue: "Export" })}
              </button>
              <Select value={importExport.exportFormat} onChange={(v) => importExport.setExportFormat(v as "json" | "csv" | "markdown")}
                data-testid="prompts-export-format"
                options={[
                  { label: "JSON", value: "json" },
                  { label: "CSV", value: "csv", disabled: !isOnline },
                  { label: "Markdown", value: "markdown", disabled: !isOnline }
                ]}
                variant="borderless" style={{ width: 120 }} popupMatchSelectWidth={false}
              />
            </div>
            {isOnline && (localSyncBatchPlan.tasks.length > 0 || sync.batchSyncState.failed.length > 0 || sync.batchSyncState.running) && (
              <Tooltip title={localSyncBatchPlan.skippedConflicts > 0
                ? t("managePrompts.sync.batchConflictHint", { defaultValue: "{{count}} conflict prompt(s) require manual resolution.", count: localSyncBatchPlan.skippedConflicts })
                : undefined}>
                <button type="button" onClick={sync.handleBatchSyncAction} data-testid="prompts-sync-all"
                  className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2 disabled:opacity-50">
                  <Cloud className="size-4" />
                  {sync.batchSyncState.running
                    ? t("managePrompts.sync.batchCancel", { defaultValue: "Cancel sync" })
                    : sync.batchSyncState.failed.length > 0
                      ? t("managePrompts.sync.batchRetryFailed", { defaultValue: "Retry failed ({{count}})", count: sync.batchSyncState.failed.length })
                      : t("managePrompts.sync.batchSyncAll", { defaultValue: "Sync all" })}
                </button>
              </Tooltip>
            )}
            <div className="inline-flex items-center rounded-md border border-border overflow-hidden">
              <button onClick={() => { if (guardPrivateMode()) return; importExport.fileInputRef.current?.click() }}
                data-testid="prompts-import"
                className="inline-flex items-center gap-2 px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2">
                <UploadCloud className="size-4" /> {t("managePrompts.import", { defaultValue: "Import" })}
              </button>
              <Select value={importExport.importMode} onChange={(v) => importExport.setImportMode(v as any)}
                data-testid="prompts-import-mode"
                options={[
                  { label: t("managePrompts.importMode.merge", { defaultValue: "Merge" }), value: "merge" },
                  { label: t("managePrompts.importMode.replaceWithBackup", { defaultValue: "Replace (backup)" }), value: "replace" }
                ]}
                variant="borderless" style={{ width: 130 }} popupMatchSelectWidth={false}
              />
            </div>
            <Tooltip title={t("managePrompts.shortcuts.openHint", { defaultValue: "Keyboard shortcuts (?)" })}>
              <button type="button" onClick={onOpenShortcutsHelp} data-testid="prompts-shortcuts-help-button"
                aria-label={t("managePrompts.shortcuts.openLabel", { defaultValue: "Open keyboard shortcuts" })}
                className="inline-flex items-center gap-2 rounded-md border border-border px-2 py-2 text-md font-medium leading-4 text-text hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2">
                <Keyboard className="size-4" aria-hidden="true" />
                {t("managePrompts.shortcuts.openButton", { defaultValue: "Shortcuts" })}
              </button>
            </Tooltip>
            <input ref={importExport.fileInputRef} type="file" accept="application/json" className="hidden"
              data-testid="prompts-import-file"
              aria-label={t("managePrompts.importFileLabel", { defaultValue: "Import prompts file" })}
              onChange={(e) => { const file = e.target.files?.[0]; if (file) importExport.handleImportFile(file); e.currentTarget.value = "" }}
            />
          </div>
          {/* Right: Filters */}
          <div className="flex w-full flex-wrap items-stretch gap-2 sm:w-auto sm:items-center sm:justify-end">
            <div data-testid="prompts-search-control" className="w-full sm:w-auto">
              <Input ref={searchInputRef} allowClear
                placeholder={t("managePrompts.searchWithScope", { defaultValue: "Search name, content, tags..." })}
                value={searchText} onChange={(e) => setSearchText(e.target.value)}
                data-testid="prompts-search"
                aria-label={t("managePrompts.search", { defaultValue: "Search prompts..." })}
                suffix={<kbd className="rounded border border-border px-1 text-xs text-text-subtle">/</kbd>}
                style={{ width: isCompactViewport ? "100%" : 260 }}
              />
            </div>
            {/* Mobile: toggle button for secondary filters */}
            {isCompactViewport && (
              <button
                type="button"
                onClick={() => setMobileFiltersExpanded((p) => !p)}
                data-testid="prompts-mobile-filters-toggle"
                aria-expanded={mobileFiltersExpanded}
                className="inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-border px-3 py-2 text-sm text-text-muted hover:bg-surface2"
              >
                <Filter className="size-4" />
                {t("managePrompts.filter.toggleFilters", { defaultValue: "Filters" })}
                {mobileFiltersExpanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
              </button>
            )}
            {/* Secondary filters — always visible on desktop, collapsible on mobile */}
            <div
              data-testid="prompts-secondary-filters"
              className={`flex w-full flex-wrap items-stretch gap-2 sm:w-auto sm:items-center sm:justify-end ${
                isCompactViewport && !mobileFiltersExpanded ? "hidden" : ""
              }`}
            >
            <div data-testid="prompts-type-filter-control" className="w-full sm:w-auto">
              <Select value={typeFilter} onChange={(v) => setTypeFilter(v as any)} data-testid="prompts-type-filter"
                aria-label={t("managePrompts.filter.typeLabel", { defaultValue: "Filter by type" })}
                style={{ width: isCompactViewport ? "100%" : 130 }}
                options={[
                  { label: t("managePrompts.filter.all", { defaultValue: "All types" }), value: "all" },
                  { label: t("managePrompts.filter.system", { defaultValue: "System" }), value: "system" },
                  { label: t("managePrompts.filter.quick", { defaultValue: "Quick" }), value: "quick" }
                ]}
              />
            </div>
            <div data-testid="prompts-usage-filter-control" className="w-full sm:w-auto">
              <Select value={usageFilter} onChange={(v) => setUsageFilter(v as "all" | "used" | "unused")}
                data-testid="prompts-usage-filter"
                aria-label={t("managePrompts.filter.usageLabel", { defaultValue: "Filter by usage" })}
                style={{ width: isCompactViewport ? "100%" : 150 }}
                options={[
                  { label: t("managePrompts.filter.usageAll", { defaultValue: "All usage" }), value: "all" },
                  { label: t("managePrompts.filter.usageUsed", { defaultValue: "Used" }), value: "used" },
                  { label: t("managePrompts.filter.usageUnused", { defaultValue: "Unused" }), value: "unused" }
                ]}
              />
            </div>
            <div data-testid="prompts-tag-filter-control" className="w-full sm:w-auto">
              <Select mode="multiple" allowClear
                placeholder={t("managePrompts.tags.placeholder", { defaultValue: "Filter by tags" })}
                style={{ width: isCompactViewport ? "100%" : 220 }}
                value={tagFilter} onChange={(v) => setTagFilter(v)}
                data-testid="prompts-tag-filter"
                aria-label={t("managePrompts.tags.filterLabel", { defaultValue: "Filter by keywords" })}
                options={allTags.map((tag) => ({ label: tag, value: tag }))}
              />
            </div>
            <div data-testid="prompts-tag-match-mode-control" className="w-full sm:w-auto">
              <Segmented value={tagMatchMode} onChange={(value) => setTagMatchMode(value as TagMatchMode)}
                size="small" data-testid="prompts-tag-match-mode"
                style={{ width: isCompactViewport ? "100%" : undefined }}
                options={[
                  { value: "any", label: t("managePrompts.tags.matchAny", { defaultValue: "Match any" }) },
                  { value: "all", label: t("managePrompts.tags.matchAll", { defaultValue: "Match all" }) }
                ]}
              />
            </div>
            {selectedSegment === "custom" && (
              <div className="w-full sm:w-auto">
                <Segmented value={viewMode} onChange={(value) => setViewMode(value as PromptViewMode)}
                  size="small" data-testid="prompts-view-mode" aria-label="View mode"
                  options={[
                    { value: "table", icon: <List className="h-3.5 w-3.5" />, label: "Table" },
                    { value: "gallery", icon: <LayoutGrid className="h-3.5 w-3.5" />, label: "Gallery" }
                  ]}
                />
              </div>
            )}
            {viewMode === "gallery" && selectedSegment === "custom" && (
              <div className="w-full sm:w-auto">
                <Segmented value={galleryDensity} onChange={(value) => setGalleryDensity(value as PromptGalleryDensity)}
                  size="small" data-testid="prompts-gallery-density" aria-label="Gallery density"
                  options={[{ value: "rich", label: "Rich" }, { value: "compact", label: "Compact" }]}
                />
              </div>
            )}
            {(viewMode === "table" || selectedSegment !== "custom") && (
              <div className="w-full sm:w-auto">
                <Segmented value={tableDensity} onChange={(value) => setTableDensity(value as PromptTableDensity)}
                  size="small" data-testid="prompts-table-density"
                  aria-label={t("managePrompts.tableDensity.label", { defaultValue: "Table density" })}
                  options={[
                    { value: "comfortable", label: t("managePrompts.tableDensity.comfortable", { defaultValue: "Comfortable" }) },
                    { value: "compact", label: t("managePrompts.tableDensity.compact", { defaultValue: "Compact" }) },
                    { value: "dense", label: t("managePrompts.tableDensity.dense", { defaultValue: "Dense" }) }
                  ]}
                />
              </div>
            )}
            </div>{/* end secondary filters wrapper */}
          </div>
        </PromptListToolbar>
      </div>

      {customPromptsLoading && <Skeleton paragraph={{ rows: 8 }} />}

      {useServerSearchResults && hiddenServerResultsOnPage > 0 && (
        <Alert type="info" showIcon className="mb-3"
          message={t("managePrompts.search.localSubset", { defaultValue: "Showing synced local matches only" })}
          description={t("managePrompts.search.localSubsetDesc", { defaultValue: "{{count}} result(s) from this page are not saved locally yet.", count: hiddenServerResultsOnPage })}
        />
      )}

      {status === "success" && Array.isArray(data) && data.length === 0 && (
        <>
          <FeatureEmptyState
            title={t("settings:managePrompts.emptyTitle", { defaultValue: "No custom prompts yet" })}
            description={t("settings:managePrompts.emptyDescription", { defaultValue: "Create reusable prompts for recurring tasks, workflows, and team conventions." })}
            examples={[
              t("settings:managePrompts.emptyExample1", { defaultValue: "Save your favorite system prompt for summaries, explanations, or translations." }),
              t("settings:managePrompts.emptyExample2", { defaultValue: "Create quick prompts for common actions like drafting emails or refining notes." })
            ]}
            primaryActionLabel={t("settings:managePrompts.emptyPrimaryCta", { defaultValue: "Create prompt" })}
            onPrimaryAction={() => editor.openFullEditor()}
          />
          <div className="mt-6">
            <h3 className="mb-3 text-sm font-medium text-text-muted">Or start with a template</h3>
            <Suspense fallback={null}>
              <PromptStarterCards onUse={(starter) => editor.openFullEditor(starter)} />
            </Suspense>
          </div>
        </>
      )}

      {status === "success" && Array.isArray(data) && data.length >= 5 && shouldShowHint("keyboard-shortcuts") && (
        <Suspense fallback={null}>
          <ContextualHint id="keyboard-shortcuts" message="Press Enter to preview, E to edit, or ? for all keyboard shortcuts."
            visible={true} onDismiss={dismissHint} onShown={markHintShown} />
        </Suspense>
      )}

      {status === "success" && Array.isArray(data) && data.length > 0 && viewMode === "table" && (
        <PromptListTable
          rows={customPromptRows} total={tableTotal} loading={customPromptsLoading}
          isOnline={isOnline} isCompactViewport={isCompactViewport}
          query={customPromptTableQuery}
          selectedIds={bulk.selectedRowKeys.map((key) => String(key))}
          onQueryChange={handleCustomPromptTableQueryChange}
          onSelectionChange={(ids) => bulk.setSelectedRowKeys(ids)}
          onRowOpen={onOpenInspector}
          onEdit={editor.handleEditPromptById}
          onToggleFavorite={editor.handleTogglePromptFavorite}
          onOpenConflictResolution={sync.openConflictResolution}
          renderActions={renderCustomPromptActions}
          renderTitleMeta={renderCustomPromptTitleMeta}
          favoriteButtonTestId={(row) => `prompt-favorite-${row.id}`}
          formatRelativeTime={formatRelativePromptTime}
          selectionDisabled={isFireFoxPrivateMode}
          columnLabels={{
            title: t("managePrompts.columns.title"),
            preview: t("managePrompts.columns.prompt"),
            tags: t("managePrompts.tags.label", { defaultValue: "Tags" }),
            updated: t("managePrompts.columns.modified", { defaultValue: "Updated" }),
            lastUsed: t("managePrompts.columns.lastUsed", { defaultValue: "Last used" }),
            status: t("managePrompts.columns.sync", { defaultValue: "Sync" }),
            actions: t("managePrompts.columns.actions"),
            author: t("managePrompts.form.author.label", { defaultValue: "Author" }),
            system: t("managePrompts.form.systemPrompt.shortLabel", { defaultValue: "System" }),
            user: t("managePrompts.form.userPrompt.shortLabel", { defaultValue: "User" }),
            unknown: t("common:unknown", { defaultValue: "Unknown" }),
            offlineStatus: t("managePrompts.sync.offlineTooltip", { defaultValue: "Sync unavailable (offline). Showing last known status." }),
            edit: t("managePrompts.tooltip.edit")
          }}
          paginationShowTotal={(total, range) =>
            t("managePrompts.pagination.summary", { defaultValue: "{{start}}-{{end}} of {{total}} prompts", start: range[0], end: range[1], total })
          }
          tableDensity={tableDensity}
        />
      )}

      {status === "success" && Array.isArray(data) && data.length > 0 && viewMode === "gallery" && (
        <div className="space-y-4" data-testid="prompts-gallery-view">
          <div className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 ${galleryDensity === "compact" ? "gap-3" : "gap-4"}`}>
            <Suspense fallback={null}>
              {customPromptRows.map((prompt) => (
                <PromptGalleryCard key={prompt.id} prompt={prompt}
                  onClick={() => onOpenInspector(prompt.id)}
                  density={galleryDensity}
                  onToggleFavorite={(next) => editor.handleTogglePromptFavorite(prompt.id, next)}
                />
              ))}
            </Suspense>
          </div>
          {tableTotal > resultsPerPage && (
            <div className="flex justify-end">
              <Pagination current={currentPage} pageSize={resultsPerPage} total={tableTotal}
                showSizeChanger pageSizeOptions={["10", "20", "50", "100"]}
                onChange={(page, pageSize) => {
                  if (pageSize !== resultsPerPage) { setResultsPerPage(pageSize); setCurrentPage(1) }
                  else { setCurrentPage(page) }
                }}
                showTotal={(total, range) =>
                  t("managePrompts.pagination.summary", { defaultValue: "{{start}}-{{end}} of {{total}} prompts", start: range[0], end: range[1], total })
                }
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
