import React from "react"
import {
  Button,
  Input,
  Table,
  Tag,
  Tooltip,
  Select,
  Checkbox,
  Skeleton,
  Pagination,
  Dropdown
} from "antd"
import type { InputRef } from "antd"
import {
  History,
  Pen,
  Trash2,
  UserCircle2,
  MessageCircle,
  Copy,
  Download,
  CheckSquare,
  Tags,
  X,
  MoreHorizontal,
  Star,
  ExternalLink,
  Clock3,
  Columns2,
  Upload as UploadIcon
} from "lucide-react"
import type { GalleryCardDensity } from "./CharacterGalleryCard"
import {
  MAX_NAME_DISPLAY_LENGTH,
  MAX_DESCRIPTION_LENGTH,
  MAX_TAG_LENGTH,
  MAX_TABLE_TAGS_DISPLAYED,
  PAGE_SIZE_OPTIONS,
  normalizePageSize,
  getCharacterVisibleTags,
  truncateText,
  type CharacterListScope,
  type TableDensity
} from "./utils"
import { withCharacterNameInLabel } from "./utils"
import { validateAndCreateImageDataUrl } from "@/utils/image-utils"
import {
  sanitizeServerErrorMessage,
  buildServerLogHint
} from "@/utils/server-error-message"
import { Alert } from "@/components/ui/primitives/Alert"
import type { TFunction } from "i18next"

const LazyCharacterGalleryCard = React.lazy(() =>
  import("./CharacterGalleryCard").then((module) => ({
    default: module.CharacterGalleryCard,
  })),
)

const LazyCharacterPreviewPopup = React.lazy(() =>
  import("./CharacterPreviewPopup").then((module) => ({
    default: module.CharacterPreviewPopup,
  })),
)

const normalizeWorldBookId = (value: string | number | undefined): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value !== "string") {
    return null
  }
  const trimmed = value.trim()
  if (!/^\d+$/.test(trimmed)) {
    return null
  }
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

export type CharacterListContentProps = {
  t: TFunction
  // data
  status: "error" | "pending" | "success"
  error: Error | null
  refetch: () => void
  data: any[] | undefined
  totalCharacters: number
  pagedGalleryData: any[]
  conversationCounts: Record<string, number> | undefined
  // view
  viewMode: "table" | "gallery"
  characterListScope: CharacterListScope
  setCharacterListScope: (v: CharacterListScope) => void
  galleryDensity: GalleryCardDensity
  tableDensity: TableDensity
  // pagination
  currentPage: number
  setCurrentPage: (v: number) => void
  pageSize: number
  setPageSize: (v: number) => void
  // sort
  sortColumn: string | null
  setSortColumn: (v: string | null) => void
  sortOrder: "ascend" | "descend" | null
  setSortOrder: (v: "ascend" | "descend" | null) => void
  // filters
  hasFilters: boolean
  searchTerm: string
  filterTags: string[]
  setFilterTags: React.Dispatch<React.SetStateAction<string[]>>
  matchAllTags: boolean
  folderFilterId: string | undefined
  selectedFolderFilterLabel: string | undefined
  creatorFilter: string | undefined
  createdFromDate: string
  createdToDate: string
  updatedFromDate: string
  updatedToDate: string
  hasConversationsOnly: boolean
  favoritesOnly: boolean
  clearFilters: () => void
  // preview
  previewCharacter: any | null
  setPreviewCharacter: (v: any | null) => void
  previewCharacterWorldBooks: any[]
  previewCharacterWorldBooksLoading: boolean
  crossNavigationContext: { launchedFromWorldBooks: boolean; focusWorldBookId?: string | number }
  // inline edit
  inlineEdit: { id: string; field: string; value: string } | null
  setInlineEdit: (v: any) => void
  inlineUpdating: boolean
  inlineEditInputRef: React.RefObject<InputRef | null>
  startInlineEdit: (record: any, field: string, trigger: Element) => void
  saveInlineEdit: () => void
  cancelInlineEdit: () => void
  // bulk ops
  selectedCharacterIds: Set<string>
  setSelectedCharacterIds: React.Dispatch<React.SetStateAction<Set<string>>>
  toggleCharacterSelection: (id: string) => void
  selectAllOnPage: () => void
  clearSelection: () => void
  selectedCount: number
  hasSelection: boolean
  allOnPageSelected: boolean
  someOnPageSelected: boolean
  handleBulkDelete: () => void
  handleBulkExport: () => void
  handleOpenCompareModal: () => void
  bulkOperationLoading: boolean
  setBulkTagModalOpen: (v: boolean) => void
  // actions
  handleChat: (record: any) => void
  handleChatInNewTab: (record: any) => Promise<void>
  preloadCharacterEditor: () => Promise<unknown>
  handleEdit: (record: any, trigger?: Element) => void
  handleDuplicate: (record: any) => void
  handleDelete: (record: any) => Promise<void>
  handleExport: (record: any, format: "json" | "png") => Promise<void>
  handleViewConversations: (record: any) => void
  handleRestoreFromTrash: (record: any) => void
  handleToggleFavorite: (record: any) => Promise<void>
  handleSetDefaultCharacter: (record: any) => Promise<void>
  handleClearDefaultCharacter: () => Promise<void>
  isDefaultCharacterRecord: (record: any) => boolean
  isCharacterFavoriteRecord: (record: any) => boolean
  isPersonaCreatePending: (record: any) => boolean
  getCreatePersonaActionLabel: (record: any) => string
  openPersonaGardenForCharacter: (record: any) => void
  createPersonaFromCharacter: (record: any) => void
  openVersionHistory: (record: any) => void
  openQuickChat: (record: any) => void
  deleting: boolean
  exporting: string | null
  // conversations
  setConversationCharacter: (v: any) => void
  setCharacterChats: (v: any[]) => void
  setChatsError: (v: string | null) => void
  setConversationsOpen: (v: boolean) => void
  // onboarding
  openCreateModal: () => void
  setShowTemplates: (v: boolean) => void
  markTemplateChooserSeen: () => void
  isImportBusy: boolean
  triggerImportPicker: () => void
  // confirm
  confirmDanger: (opts: {
    title: string
    content: string
    okText: string
    cancelText: string
  }) => Promise<boolean>
}

export const CharacterListContent: React.FC<CharacterListContentProps> = (props) => {
  const {
    t,
    status,
    error,
    refetch,
    data,
    totalCharacters,
    pagedGalleryData,
    conversationCounts,
    viewMode,
    characterListScope,
    setCharacterListScope,
    galleryDensity,
    tableDensity,
    currentPage,
    setCurrentPage,
    pageSize,
    setPageSize,
    sortColumn,
    setSortColumn,
    sortOrder,
    setSortOrder,
    hasFilters,
    searchTerm,
    filterTags,
    setFilterTags,
    matchAllTags,
    folderFilterId,
    selectedFolderFilterLabel,
    creatorFilter,
    createdFromDate,
    createdToDate,
    updatedFromDate,
    updatedToDate,
    hasConversationsOnly,
    favoritesOnly,
    clearFilters,
    previewCharacter,
    setPreviewCharacter,
    previewCharacterWorldBooks,
    previewCharacterWorldBooksLoading,
    crossNavigationContext,
    inlineEdit,
    setInlineEdit,
    inlineUpdating,
    inlineEditInputRef,
    startInlineEdit,
    saveInlineEdit,
    cancelInlineEdit,
    selectedCharacterIds,
    setSelectedCharacterIds,
    toggleCharacterSelection,
    selectAllOnPage,
    clearSelection,
    selectedCount,
    hasSelection,
    allOnPageSelected,
    someOnPageSelected,
    handleBulkDelete,
    handleBulkExport,
    handleOpenCompareModal,
    bulkOperationLoading,
    setBulkTagModalOpen,
    handleChat,
    handleChatInNewTab,
    preloadCharacterEditor,
    handleEdit,
    handleDuplicate,
    handleDelete,
    handleExport,
    handleViewConversations,
    handleRestoreFromTrash,
    handleToggleFavorite,
    handleSetDefaultCharacter,
    handleClearDefaultCharacter,
    isDefaultCharacterRecord,
    isCharacterFavoriteRecord,
    isPersonaCreatePending,
    getCreatePersonaActionLabel,
    openPersonaGardenForCharacter,
    createPersonaFromCharacter,
    openVersionHistory,
    openQuickChat,
    deleting,
    exporting,
    setConversationCharacter,
    setCharacterChats,
    setChatsError,
    setConversationsOpen,
    openCreateModal,
    setShowTemplates,
    markTemplateChooserSeen,
    isImportBusy,
    triggerImportPicker,
    confirmDanger
  } = props

  const resolveTimestamp = (
    record: Record<string, any>,
    keys: string[]
  ): number | null => {
    for (const key of keys) {
      const raw = record?.[key]
      if (!raw) continue
      const timestamp =
        typeof raw === "number" ? raw : new Date(String(raw)).getTime()
      if (Number.isFinite(timestamp)) return timestamp
    }
    return null
  }

  const formatRelativeActivityTime = (timestamp: number | null): string => {
    if (!timestamp) {
      return t("settings:manageCharacters.activity.never", {
        defaultValue: "Never"
      })
    }
    const delta = Date.now() - timestamp
    const absDelta = Math.abs(delta)
    const minute = 60_000
    const hour = 3_600_000
    const day = 86_400_000
    const week = 604_800_000
    const month = 2_629_746_000
    const year = 31_556_952_000

    if (absDelta < minute) {
      return t("settings:manageCharacters.activity.justNow", {
        defaultValue: "Just now"
      })
    }

    if (absDelta < hour) {
      return t("settings:manageCharacters.activity.minutesAgo", {
        defaultValue: "{{count}}m ago",
        count: Math.max(1, Math.round(absDelta / minute))
      })
    }

    if (absDelta < day) {
      return t("settings:manageCharacters.activity.hoursAgo", {
        defaultValue: "{{count}}h ago",
        count: Math.max(1, Math.round(absDelta / hour))
      })
    }

    if (absDelta < week) {
      return t("settings:manageCharacters.activity.daysAgo", {
        defaultValue: "{{count}}d ago",
        count: Math.max(1, Math.round(absDelta / day))
      })
    }

    if (absDelta < month) {
      return t("settings:manageCharacters.activity.weeksAgo", {
        defaultValue: "{{count}}w ago",
        count: Math.max(1, Math.round(absDelta / week))
      })
    }

    if (absDelta < year) {
      return t("settings:manageCharacters.activity.monthsAgo", {
        defaultValue: "{{count}}mo ago",
        count: Math.max(1, Math.round(absDelta / month))
      })
    }

    return t("settings:manageCharacters.activity.yearsAgo", {
      defaultValue: "{{count}}y ago",
      count: Math.max(1, Math.round(absDelta / year))
    })
  }

  const formatAbsoluteActivityTime = (timestamp: number | null): string => {
    if (!timestamp) {
      return t("settings:manageCharacters.activity.never", {
        defaultValue: "Never"
      })
    }
    try {
      return new Date(timestamp).toLocaleString()
    } catch {
      return String(timestamp)
    }
  }

  return (
    <>
      {/* Accessible live region for search results */}
      <div
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        role="status"
      >
        {status === "success" &&
          t("settings:manageCharacters.aria.searchResults", {
            defaultValue: "{{count}} characters found",
            count: totalCharacters
          })}
      </div>
      {status === "error" && (
        <div className="rounded-lg border border-danger/30 bg-danger/10 p-4">
          <Alert
            variant="error"
            title={t("settings:manageCharacters.loadError.title", {
              defaultValue: "Couldn't load characters"
            })}
            className="border-0 bg-transparent p-0"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="text-sm text-danger">
                <p>
                  {sanitizeServerErrorMessage(
                    error,
                    t("settings:manageCharacters.loadError.description", {
                      defaultValue: "Check your connection and try again."
                    })
                  )}
                </p>
                <p className="mt-1 text-xs text-text-muted">
                  {buildServerLogHint(
                    error,
                    t("settings:manageCharacters.loadError.logHint", {
                      defaultValue:
                        "If the issue persists, check server logs for more details."
                    })
                  )}
                </p>
              </div>
              <Button size="small" onClick={() => refetch()}>
                {t("common:retry", { defaultValue: "Retry" })}
              </Button>
            </div>
          </Alert>
        </div>
      )}
      {status === "pending" && <Skeleton active paragraph={{ rows: 6 }} />}
      {status === "success" &&
        Array.isArray(data) &&
        data.length === 0 &&
        characterListScope === "active" &&
        !hasFilters && (
          <div className="space-y-4">
            <div className="text-center">
              <UserCircle2 className="mx-auto h-10 w-10 text-text-muted" />
              <h3 className="mt-2 text-base font-semibold">
                {t("settings:manageCharacters.emptyTitle", {
                  defaultValue: "Get started with your first character"
                })}
              </h3>
              <p className="mt-1 text-sm text-text-muted">
                {t("settings:manageCharacters.emptyDescription", {
                  defaultValue:
                    "Create reusable personas you can chat with. Each character keeps its own conversation history."
                })}
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {/* Create from scratch */}
              <button
                type="button"
                className="flex flex-col items-center gap-2 rounded-xl border border-border bg-surface p-4 text-center transition-all hover:border-primary hover:shadow-md"
                onClick={openCreateModal}
              >
                <Pen className="h-7 w-7 text-primary" />
                <span className="text-sm font-medium">
                  {t("settings:manageCharacters.onboarding.createTitle", {
                    defaultValue: "Create from scratch"
                  })}
                </span>
                <span className="text-xs text-text-muted">
                  {t("settings:manageCharacters.onboarding.createDesc", {
                    defaultValue: "Build a custom character with your own persona and system prompt"
                  })}
                </span>
              </button>

              {/* Start from a template */}
              <button
                type="button"
                className="flex flex-col items-center gap-2 rounded-xl border border-border bg-surface p-4 text-center transition-all hover:border-primary hover:shadow-md"
                onClick={() => {
                  setShowTemplates(true)
                  markTemplateChooserSeen()
                  openCreateModal()
                }}
              >
                <Copy className="h-7 w-7 text-primary" />
                <span className="text-sm font-medium">
                  {t("settings:manageCharacters.onboarding.templateTitle", {
                    defaultValue: "Start from a template"
                  })}
                </span>
                <span className="text-xs text-text-muted">
                  {t("settings:manageCharacters.onboarding.templateDesc", {
                    defaultValue: "Choose from pre-built personas like Writing Coach or Research Helper"
                  })}
                </span>
              </button>

              {/* Import existing */}
              <button
                type="button"
                className="flex flex-col items-center gap-2 rounded-xl border border-border bg-surface p-4 text-center transition-all hover:border-primary hover:shadow-md"
                disabled={isImportBusy}
                onClick={triggerImportPicker}
              >
                <UploadIcon className="h-7 w-7 text-primary" />
                <span className="text-sm font-medium">
                  {t("settings:manageCharacters.onboarding.importTitle", {
                    defaultValue: "Import existing"
                  })}
                </span>
                <span className="text-xs text-text-muted">
                  {t("settings:manageCharacters.onboarding.importDesc", {
                    defaultValue: "Upload JSON, PNG, or YAML character card files"
                  })}
                </span>
              </button>
            </div>

            <p className="text-center text-xs text-text-muted">
              {t("settings:manageCharacters.onboarding.tip", {
                defaultValue: "Tip: Characters appear in the chat header dropdown so you can quickly switch personas across conversations."
              })}
            </p>
          </div>
        )}
      {status === "success" &&
        Array.isArray(data) &&
        data.length === 0 &&
        characterListScope === "deleted" &&
        !hasFilters && (
          <div className="rounded-lg border border-dashed border-border bg-surface p-4 text-sm text-text">
            <div className="flex flex-col gap-2">
              <span className="font-medium">
                {t("settings:manageCharacters.deletedEmptyTitle", {
                  defaultValue: "No recently deleted characters"
                })}
              </span>
              <span className="text-text-muted">
                {t("settings:manageCharacters.deletedEmptyDescription", {
                  defaultValue:
                    "Soft-deleted characters appear here while they remain within the server restore window."
                })}
              </span>
              <div>
                <Button
                  size="small"
                  onClick={() => setCharacterListScope("active")}>
                  {t("settings:manageCharacters.scope.backToActive", {
                    defaultValue: "Back to active"
                  })}
                </Button>
              </div>
            </div>
          </div>
        )}
      {status === "success" &&
        Array.isArray(data) &&
        data.length === 0 &&
        hasFilters && (
          <div className="rounded-lg border border-dashed border-border bg-surface p-4 text-sm text-text">
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span>
                  {t("settings:manageCharacters.filteredEmptyTitle", {
                    defaultValue: "No characters match your filters"
                  })}
                </span>
                <Button
                  size="small"
                  onClick={() => {
                    clearFilters()
                    refetch()
                  }}>
                  {t("settings:manageCharacters.filter.clear", {
                    defaultValue: "Clear filters"
                  })}
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs text-text-subtle">
                {searchTerm.trim() && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeSearch", {
                      defaultValue: "Search: \"{{term}}\"",
                      term: searchTerm.trim()
                    })}
                  </span>
                )}
                {filterTags.length > 0 && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeTags", {
                      defaultValue: "Tags: {{tags}}",
                      tags: filterTags.join(", ")
                    })}
                    {matchAllTags && (
                      <span className="text-text-subtle">
                        ({t("settings:manageCharacters.filter.matchAllLabel", { defaultValue: "all" })})
                      </span>
                    )}
                  </span>
                )}
                {selectedFolderFilterLabel && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeFolder", {
                      defaultValue: "Folder: {{folder}}",
                      folder: selectedFolderFilterLabel
                    })}
                  </span>
                )}
                {creatorFilter && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeCreator", {
                      defaultValue: "Creator: {{creator}}",
                      creator: creatorFilter
                    })}
                  </span>
                )}
                {(createdFromDate || createdToDate) && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeCreatedRange", {
                      defaultValue: "Created: {{from}} to {{to}}",
                      from: createdFromDate || "\u2014",
                      to: createdToDate || "\u2014"
                    })}
                  </span>
                )}
                {(updatedFromDate || updatedToDate) && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeUpdatedRange", {
                      defaultValue: "Updated: {{from}} to {{to}}",
                      from: updatedFromDate || "\u2014",
                      to: updatedToDate || "\u2014"
                    })}
                  </span>
                )}
                {hasConversationsOnly && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeHasConversations", {
                      defaultValue: "Has conversations"
                    })}
                  </span>
                )}
                {favoritesOnly && (
                  <span className="inline-flex items-center gap-1 rounded bg-surface2 px-2 py-0.5">
                    {t("settings:manageCharacters.filter.activeFavoritesOnly", {
                      defaultValue: "Favorites only"
                    })}
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
      {status === "success" && Array.isArray(data) && data.length > 0 && viewMode === 'table' && (
        <div className="space-y-3">
          {characterListScope === "deleted" && (
            <div className="rounded-md border border-border bg-surface2 p-3 text-sm text-text-muted">
              {t("settings:manageCharacters.deletedListDescription", {
                defaultValue:
                  "Showing recently deleted characters. Restore is available while each item remains within the server restore window."
              })}
            </div>
          )}
          {/* Bulk Actions Toolbar (M5) */}
          {characterListScope === "active" && hasSelection && (
            <div className="flex items-center gap-3 p-2 bg-surface rounded-lg border border-border">
              <div className="flex items-center gap-2">
                <CheckSquare className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">
                  {t("settings:manageCharacters.bulk.selected", {
                    defaultValue: "{{count}} selected",
                    count: selectedCount
                  })}
                </span>
              </div>
              <div className="flex items-center gap-2 ml-auto">
                <Tooltip title={t("settings:manageCharacters.bulk.addTags", { defaultValue: "Add tags" })}>
                  <Button
                    size="small"
                    icon={<Tags className="w-4 h-4" />}
                    onClick={() => setBulkTagModalOpen(true)}
                    loading={bulkOperationLoading}>
                    {t("settings:manageCharacters.bulk.addTags", { defaultValue: "Add tags" })}
                  </Button>
                </Tooltip>
                <Tooltip
                  title={
                    selectedCount === 2
                      ? t("settings:manageCharacters.bulk.compareReady", {
                          defaultValue: "Compare selected characters"
                        })
                      : t("settings:manageCharacters.bulk.compareHint", {
                          defaultValue: "Select exactly 2 characters to compare"
                        })
                  }>
                  <Button
                    size="small"
                    icon={<Columns2 className="w-4 h-4" />}
                    onClick={handleOpenCompareModal}
                    disabled={selectedCount !== 2 || bulkOperationLoading}>
                    {t("settings:manageCharacters.bulk.compare", {
                      defaultValue: "Compare"
                    })}
                  </Button>
                </Tooltip>
                <Tooltip title={t("settings:manageCharacters.bulk.export", { defaultValue: "Export" })}>
                  <Button
                    size="small"
                    icon={<Download className="w-4 h-4" />}
                    onClick={handleBulkExport}
                    loading={bulkOperationLoading}>
                    {t("settings:manageCharacters.bulk.export", { defaultValue: "Export" })}
                  </Button>
                </Tooltip>
                <Tooltip title={t("settings:manageCharacters.bulk.delete", { defaultValue: "Delete" })}>
                  <Button
                    size="small"
                    danger
                    icon={<Trash2 className="w-4 h-4" />}
                    onClick={handleBulkDelete}
                    loading={bulkOperationLoading}>
                    {t("settings:manageCharacters.bulk.delete", { defaultValue: "Delete" })}
                  </Button>
                </Tooltip>
                <Tooltip title={t("settings:manageCharacters.bulk.clearSelection", { defaultValue: "Clear selection" })}>
                  <Button
                    size="small"
                    type="text"
                    icon={<X className="w-4 h-4" />}
                    onClick={clearSelection}
                    aria-label={t("settings:manageCharacters.bulk.clearSelection", { defaultValue: "Clear selection" })}
                  />
                </Tooltip>
              </div>
            </div>
          )}
          <div className="overflow-x-auto" data-testid="characters-table-view">
            <Table
              className={`characters-table-density-${tableDensity}`}
              size={tableDensity === "comfortable" ? "middle" : "small"}
              rowKey={(r: any) => r.id || r.slug || r.name}
              dataSource={data}
              onRow={(record) => ({
                onClick: (event) => {
                  if (characterListScope !== "active") return
                  const target = event.target as HTMLElement | null
                  if (
                    target?.closest(
                      "button, a, input, textarea, select, [role='button'], [role='menuitem'], .ant-select, .ant-dropdown, .ant-checkbox-wrapper"
                    )
                  ) {
                    return
                  }
                  setPreviewCharacter(record)
                }
              })}
              pagination={{
                current: currentPage,
                pageSize,
                total: totalCharacters,
                showSizeChanger: true,
                pageSizeOptions: PAGE_SIZE_OPTIONS.map((size) => String(size)),
                onShowSizeChange: (_page, nextPageSize) => {
                  setPageSize(normalizePageSize(nextPageSize))
                  setCurrentPage(1)
                },
                onChange: (page, nextPageSize) => {
                  const normalizedNextPageSize = normalizePageSize(nextPageSize)
                  if (normalizedNextPageSize !== pageSize) {
                    setPageSize(normalizedNextPageSize)
                    setCurrentPage(1)
                    return
                  }
                  setCurrentPage(page)
                }
              }}
              onChange={(_pagination, _filters, sorter) => {
                // Handle sort state for persistence
                if (!Array.isArray(sorter)) {
                  const nextOrder = sorter.order || null
                  setSortOrder(nextOrder)
                  setSortColumn(nextOrder ? ((sorter.columnKey as string) || null) : null)
                }
              }}
              columns={[
              characterListScope === "active" ? {
                // Bulk selection checkbox column (M5)
                title: (
                  <Checkbox
                    checked={allOnPageSelected}
                    indeterminate={someOnPageSelected}
                    onChange={(e) => {
                      if (e.target.checked) {
                        selectAllOnPage()
                      } else {
                        // Deselect all on current page
                        if (!Array.isArray(data)) return
                        const pageIds = data.map((c: any) => String(c.id || c.slug || c.name))
                        setSelectedCharacterIds((prev) => {
                          const next = new Set(prev)
                          pageIds.forEach((id) => next.delete(id))
                          return next
                        })
                      }
                    }}
                    aria-label={t("settings:manageCharacters.bulk.selectAll", { defaultValue: "Select all on page" })}
                  />
                ),
                key: "selection",
                width: 48,
                render: (_: any, record: any) => {
                  const recordId = String(record.id || record.slug || record.name)
                  return (
                    <Checkbox
                      checked={selectedCharacterIds.has(recordId)}
                      onChange={(e) => {
                        e.stopPropagation()
                        toggleCharacterSelection(recordId)
                      }}
                      aria-label={t("settings:manageCharacters.bulk.selectOne", {
                        defaultValue: "Select {{name}}",
                        name: record.name || recordId
                      })}
                    />
                  )
                }
              } : null,
              {
                title: t("settings:manageCharacters.columns.name", {
                  defaultValue: "Name"
                }),
                dataIndex: "name",
                key: "name",
                sorter: true,
                sortDirections: ["ascend", "descend"] as const,
                sortOrder: sortColumn === "name" ? sortOrder : undefined,
                width: 360,
                render: (v: string, record: any) => {
                  const recordId = String(record.id || record.slug || record.name)
                  const isNameEditing =
                    inlineEdit?.id === recordId && inlineEdit?.field === "name"
                  const isDescriptionEditing =
                    inlineEdit?.id === recordId && inlineEdit?.field === "description"
                  const descriptionValue = String(record?.description || "").trim()
                  const descriptionLineClass =
                    tableDensity === "comfortable" ? "line-clamp-2" : "line-clamp-1"
                  const count = conversationCounts?.[recordId] || 0

                  const avatarSrc =
                    record?.avatar_url ||
                    validateAndCreateImageDataUrl(record?.image_base64)

                  const avatarNode = avatarSrc ? (
                    <img
                      src={avatarSrc}
                      loading="lazy"
                      decoding="async"
                      className="h-6 w-6 rounded-full object-cover"
                      alt={
                        record?.name
                          ? t("settings:manageCharacters.avatarAltWithName", {
                              defaultValue: "Avatar of {{name}}",
                              name: record.name
                            })
                          : t("settings:manageCharacters.avatarAlt", {
                              defaultValue: "User avatar"
                            })
                      }
                    />
                  ) : (
                    <UserCircle2 className="h-5 w-5" />
                  )

                  const nameNode =
                    characterListScope === "deleted" ? (
                      <span className="line-clamp-1 font-medium" title={v || undefined}>
                        {truncateText(v, MAX_NAME_DISPLAY_LENGTH)}
                      </span>
                    ) : isNameEditing ? (
                      <Input
                        ref={inlineEditInputRef}
                        size="small"
                        value={inlineEdit.value}
                        onChange={(e) => setInlineEdit({ ...inlineEdit, value: e.target.value })}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault()
                            saveInlineEdit()
                          } else if (e.key === "Escape") {
                            cancelInlineEdit()
                          }
                        }}
                        onBlur={saveInlineEdit}
                        disabled={inlineUpdating}
                        className="max-w-[240px]"
                      />
                    ) : (
                      <Tooltip
                        title={t("settings:manageCharacters.table.doubleClickEdit", {
                          defaultValue: "Double-click to edit"
                        })}
                      >
                        <span
                          className="group/edit inline-flex items-center gap-1 line-clamp-1 cursor-text rounded px-1 -mx-1 font-medium hover:bg-surface-hover"
                          title={v || undefined}
                          data-inline-edit-key={`${recordId}:name`}
                          role="button"
                          tabIndex={0}
                          aria-label={withCharacterNameInLabel(
                            t("settings:manageCharacters.table.inlineEditName", {
                              defaultValue: "Edit name inline for {{name}}",
                              name: v || record?.name || record?.slug || "character"
                            }),
                            "Edit name inline for {{name}}",
                            String(v || record?.name || record?.slug || "character")
                          )}
                          onDoubleClick={(event) =>
                            startInlineEdit(record, "name", event.currentTarget)
                          }
                          onKeyDown={(event) => {
                            if (
                              event.key === "Enter" ||
                              event.key === "F2" ||
                              event.key === " " ||
                              event.key === "Spacebar"
                            ) {
                              event.preventDefault()
                              startInlineEdit(record, "name", event.currentTarget)
                            }
                          }}
                        >
                          {truncateText(v, MAX_NAME_DISPLAY_LENGTH)}
                          <Pen className="h-3 w-3 flex-shrink-0 opacity-0 transition-opacity group-hover/edit:opacity-40" />
                        </span>
                      </Tooltip>
                    )

                  const descriptionNode = isDescriptionEditing ? (
                    <Input
                      ref={inlineEditInputRef}
                      size="small"
                      value={inlineEdit.value}
                      onChange={(e) => setInlineEdit({ ...inlineEdit, value: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault()
                          saveInlineEdit()
                        } else if (e.key === "Escape") {
                          cancelInlineEdit()
                        }
                      }}
                      onBlur={saveInlineEdit}
                      disabled={inlineUpdating}
                      className="max-w-[320px]"
                    />
                  ) : descriptionValue ? (
                    <Tooltip
                      title={t("settings:manageCharacters.table.doubleClickEdit", {
                        defaultValue: "Double-click to edit"
                      })}
                    >
                      <span
                        className={`group/edit-desc inline-flex items-center gap-1 ${descriptionLineClass} cursor-text rounded px-1 -mx-1 text-xs text-text-muted hover:bg-surface-hover`}
                        title={descriptionValue}
                        data-inline-edit-key={`${recordId}:description`}
                        role="button"
                        tabIndex={0}
                        aria-label={t("settings:manageCharacters.table.inlineEditDescription", {
                          defaultValue: "Edit description inline for {{name}}",
                          name: record?.name || record?.slug || "character"
                        })}
                        onDoubleClick={(event) =>
                          startInlineEdit(record, "description", event.currentTarget)
                        }
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === "F2" ||
                            event.key === " " ||
                            event.key === "Spacebar"
                          ) {
                            event.preventDefault()
                            startInlineEdit(record, "description", event.currentTarget)
                          }
                        }}
                      >
                        {truncateText(descriptionValue, MAX_DESCRIPTION_LENGTH * 2)}
                        <Pen className="h-3 w-3 flex-shrink-0 opacity-0 transition-opacity group-hover/edit-desc:opacity-40" />
                      </span>
                    </Tooltip>
                  ) : null

                  return (
                    <div className="flex min-w-0 items-start gap-2">
                      <div className="pt-0.5">{avatarNode}</div>
                      <div className="min-w-0 flex-1">
                        <div className="flex min-w-0 items-center gap-2">
                          <div className="min-w-0 flex-1">{nameNode}</div>
                          {count > 0 && (
                            <Tooltip
                              title={t(
                                "settings:manageCharacters.gallery.conversationCount",
                                {
                                  defaultValue: "{{count}} conversation(s)",
                                  count
                                }
                              )}
                            >
                              <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                                <MessageCircle className="h-3 w-3" />
                                {count > 99 ? "99+" : count}
                              </span>
                            </Tooltip>
                          )}
                        </div>
                        {descriptionNode && <div className="mt-0.5 min-w-0">{descriptionNode}</div>}
                      </div>
                    </div>
                  )
                }
              },
              {
                title: t("settings:manageCharacters.tags.label", {
                  defaultValue: "Tags"
                }),
                dataIndex: "tags",
                key: "tags",
                width: 220,
                render: (tags: unknown) => {
                  const all = getCharacterVisibleTags(tags)
                  if (all.length === 0) return null
                  const visible = all.slice(0, MAX_TABLE_TAGS_DISPLAYED)
                  const hasMore = all.length > MAX_TABLE_TAGS_DISPLAYED
                  const hiddenCount = all.length - MAX_TABLE_TAGS_DISPLAYED
                  const hiddenTags = all.slice(MAX_TABLE_TAGS_DISPLAYED)
                  const applyTagFilterSelection = (tagsToAdd: string[]) => {
                    if (tagsToAdd.length === 0) return
                    setFilterTags((prev) => {
                      const next = [...prev]
                      let changed = false
                      for (const candidateTag of tagsToAdd) {
                        if (!next.includes(candidateTag)) {
                          next.push(candidateTag)
                          changed = true
                        }
                      }
                      return changed ? next : prev
                    })
                    setCurrentPage(1)
                  }
                  return (
                    <div className="flex min-w-0 flex-wrap items-center gap-1">
                      {visible.map((tag: string, index: number) => (
                        <button
                          key={`${tag}-${index}`}
                          type="button"
                          className="rounded-sm"
                          onClick={(event) => {
                            event.stopPropagation()
                            applyTagFilterSelection([tag])
                          }}
                        >
                          <Tag>{truncateText(tag, MAX_TAG_LENGTH)}</Tag>
                        </button>
                      ))}
                      {hasMore && (
                        <Tooltip
                          trigger={["hover", "focus"]}
                          title={
                            <div>
                              <div className="mb-1 font-medium">
                                {t("settings:manageCharacters.tags.moreCount", {
                                  defaultValue: "+{{count}} more tags",
                                  count: hiddenCount
                                })}
                              </div>
                              <div className="text-xs">{hiddenTags.join(", ")}</div>
                            </div>
                          }
                        >
                          <button
                            type="button"
                            className="cursor-help rounded-sm text-xs text-text-subtle underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
                            aria-label={t("settings:manageCharacters.tags.hiddenTagsAriaLabel", {
                              defaultValue: "Show and filter +{{count}} hidden tags",
                              count: hiddenCount
                            })}
                            onClick={(event) => {
                              event.stopPropagation()
                              applyTagFilterSelection(hiddenTags)
                            }}
                            onKeyDown={(event) => {
                              if (event.key !== "Enter" && event.key !== " ") return
                              event.preventDefault()
                              event.stopPropagation()
                              applyTagFilterSelection(hiddenTags)
                            }}
                          >
                            +{hiddenCount}
                          </button>
                        </Tooltip>
                      )}
                    </div>
                  )
                }
              },
              {
                title: t("settings:manageCharacters.columns.creator", {
                  defaultValue: "Creator"
                }),
                key: "creator",
                sorter: true,
                sortDirections: ["ascend", "descend"] as const,
                sortOrder: sortColumn === "creator" ? sortOrder : undefined,
                width: 180,
                render: (_: any, record: any) => {
                  const creatorValue =
                    record.creator || record.created_by || record.createdBy
                  return creatorValue ? (
                    <span className="line-clamp-1 text-xs text-text">{creatorValue}</span>
                  ) : null
                }
              },
              {
                title: t("settings:manageCharacters.columns.activity", {
                  defaultValue: "Activity"
                }),
                key: "activity",
                sorter: true,
                sortDirections: ["ascend", "descend"] as const,
                sortOrder:
                  sortColumn === "activity" || sortColumn === "lastUsedAt"
                    ? sortOrder
                    : undefined,
                width: 180,
                render: (_: any, record: any) => {
                  const lastUsedTimestamp = resolveTimestamp(record, [
                    "last_used_at",
                    "lastUsedAt",
                    "last_active",
                    "lastActive"
                  ])
                  const updatedTimestamp = resolveTimestamp(record, [
                    "updated_at",
                    "updatedAt",
                    "modified_at",
                    "modifiedAt"
                  ])
                  const createdTimestamp = resolveTimestamp(record, [
                    "created_at",
                    "createdAt",
                    "created"
                  ])
                  const activityTimestamp =
                    lastUsedTimestamp ?? updatedTimestamp ?? createdTimestamp
                  const secondaryText = updatedTimestamp
                    ? t("settings:manageCharacters.activity.updatedSecondary", {
                        defaultValue: "Updated {{time}}",
                        time: formatRelativeActivityTime(updatedTimestamp)
                      })
                    : createdTimestamp
                      ? t("settings:manageCharacters.activity.createdSecondary", {
                          defaultValue: "Created {{time}}",
                          time: formatRelativeActivityTime(createdTimestamp)
                        })
                      : ""

                  return (
                    <Tooltip
                      placement="topLeft"
                      title={
                        <div className="space-y-1 text-xs">
                          <div>
                            {t("settings:manageCharacters.columns.lastUsedAt", {
                              defaultValue: "Last used"
                            })}
                            : {formatAbsoluteActivityTime(lastUsedTimestamp)}
                          </div>
                          <div>
                            {t("settings:manageCharacters.columns.updatedAt", {
                              defaultValue: "Updated"
                            })}
                            : {formatAbsoluteActivityTime(updatedTimestamp)}
                          </div>
                          <div>
                            {t("settings:manageCharacters.columns.createdAt", {
                              defaultValue: "Created"
                            })}
                            : {formatAbsoluteActivityTime(createdTimestamp)}
                          </div>
                        </div>
                      }
                    >
                      <div className="min-w-0">
                        <div className="text-xs font-medium text-text">
                          {formatRelativeActivityTime(activityTimestamp)}
                        </div>
                        {secondaryText && (
                          <div className="line-clamp-1 text-[11px] text-text-subtle">
                            {secondaryText}
                          </div>
                        )}
                      </div>
                    </Tooltip>
                  )
                }
              },
            {
              title: t("settings:manageCharacters.columns.actions", {
                defaultValue: "Actions"
              }),
              key: "actions",
              width: 210,
              render: (_: any, record: any) => {
                const chatLabel = t("settings:manageCharacters.actions.chat", {
                  defaultValue: "Chat"
                })
                const editLabel = t(
                  "settings:manageCharacters.actions.edit",
                  {
                    defaultValue: "Edit"
                  }
                )
                const deleteLabel = t(
                  "settings:manageCharacters.actions.delete",
                  {
                    defaultValue: "Delete"
                  }
                )
                const duplicateLabel = t(
                  "settings:manageCharacters.actions.duplicate",
                  {
                    defaultValue: "Duplicate"
                  }
                )
                const addFavoriteLabel = t(
                  "settings:manageCharacters.actions.addFavorite",
                  {
                    defaultValue: "Add favorite"
                  }
                )
                const removeFavoriteLabel = t(
                  "settings:manageCharacters.actions.removeFavorite",
                  {
                    defaultValue: "Remove favorite"
                  }
                )
                const restoreLabel = t(
                  "settings:manageCharacters.actions.restore",
                  {
                    defaultValue: "Restore"
                  }
                )
                const name = record?.name || record?.title || record?.slug || ""
                const isDefaultCharacter = isDefaultCharacterRecord(record)
                const isFavorite = isCharacterFavoriteRecord(record)

                if (characterListScope === "deleted") {
                  return (
                    <div className="flex items-center gap-1 whitespace-nowrap">
                      <Tooltip title={restoreLabel}>
                        <button
                          type="button"
                          className="inline-flex items-center rounded-md border border-transparent p-1.5 text-primary transition motion-reduce:transition-none hover:border-primary/30 hover:bg-primary/10 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg"
                          aria-label={withCharacterNameInLabel(
                            t("settings:manageCharacters.aria.restore", {
                              defaultValue: "Restore character {{name}}",
                              name
                            }),
                            "Restore character {{name}}",
                            name
                          )}
                          onClick={() => handleRestoreFromTrash(record)}>
                          <History className="w-4 h-4" />
                        </button>
                      </Tooltip>
                    </div>
                  )
                }
                return (
                  <div className="flex items-center gap-1 whitespace-nowrap">
                    {/* Primary: Chat */}
                    <Tooltip
                      title={chatLabel}>
                      <button
                        type="button"
                        className="inline-flex items-center rounded-md border border-transparent p-1.5 text-primary transition motion-reduce:transition-none hover:border-primary/30 hover:bg-primary/10 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg"
                        aria-label={withCharacterNameInLabel(
                          t("settings:manageCharacters.aria.chatWith", {
                            defaultValue: "Chat as {{name}}",
                            name
                          }),
                          "Chat as {{name}}",
                          name
                        )}
                        onClick={() => handleChat(record)}>
                        <MessageCircle className="w-4 h-4" />
                      </button>
                    </Tooltip>
                    {/* Primary: Edit */}
                    <Tooltip
                      title={editLabel}>
                      <button
                        type="button"
                        className="inline-flex items-center rounded-md border border-transparent p-1.5 text-text-muted transition motion-reduce:transition-none hover:border-border hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg"
                        aria-label={withCharacterNameInLabel(
                          t("settings:manageCharacters.aria.edit", {
                            defaultValue: "Edit character {{name}}",
                            name
                          }),
                          "Edit character {{name}}",
                          name
                        )}
                        onMouseEnter={() => {
                          void preloadCharacterEditor()
                        }}
                        onFocus={() => {
                          void preloadCharacterEditor()
                        }}
                        onClick={(e) => {
                          handleEdit(record, e.currentTarget)
                        }}>
                        <Pen className="w-4 h-4" />
                      </button>
                    </Tooltip>
                    {/* Primary: Delete */}
                    <Tooltip
                      title={deleteLabel}>
                      <button
                        type="button"
                        className="inline-flex items-center rounded-md border border-transparent p-1.5 text-danger transition motion-reduce:transition-none hover:border-danger/30 hover:bg-danger/10 focus:outline-none focus:ring-2 focus:ring-danger focus:ring-offset-1 focus:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-60"
                        aria-label={withCharacterNameInLabel(
                          t("settings:manageCharacters.aria.delete", {
                            defaultValue: "Delete character {{name}}",
                            name
                          }),
                          "Delete character {{name}}",
                          name
                        )}
                        disabled={deleting}
                        onClick={async () => {
                          const ok = await confirmDanger({
                            title: t("common:confirmTitle", {
                              defaultValue: "Please confirm"
                            }),
                            content: t(
                              "settings:manageCharacters.confirm.delete",
                              {
                                defaultValue:
                                  "Are you sure you want to delete this character? It will be soft-deleted and can be undone for 10 seconds."
                              }
                            ),
                            okText: t("common:delete", { defaultValue: "Delete" }),
                            cancelText: t("common:cancel", {
                              defaultValue: "Cancel"
                            })
                          })
                          if (ok) {
                            await handleDelete(record)
                          }
                        }}>
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </Tooltip>
                    <Tooltip
                      title={isFavorite ? removeFavoriteLabel : addFavoriteLabel}>
                      <button
                        type="button"
                        className={`inline-flex items-center rounded-md border border-transparent p-1.5 transition motion-reduce:transition-none focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg ${
                          isFavorite
                            ? "text-primary hover:border-primary/30 hover:bg-primary/10"
                            : "text-text-muted hover:border-border hover:bg-surface2"
                        }`}
                        aria-label={withCharacterNameInLabel(
                          t(
                            isFavorite
                              ? "settings:manageCharacters.aria.removeFavorite"
                              : "settings:manageCharacters.aria.addFavorite",
                            {
                              defaultValue: isFavorite
                                ? "Remove {{name}} from favorites"
                                : "Add {{name}} to favorites",
                              name
                            }
                          ),
                          isFavorite
                            ? "Remove {{name}} from favorites"
                            : "Add {{name}} to favorites",
                          name
                        )}
                        onClick={() => {
                          void handleToggleFavorite(record)
                        }}>
                        {isFavorite ? (
                          <Star className="w-4 h-4 fill-current" />
                        ) : (
                          <Star className="w-4 h-4" />
                        )}
                      </button>
                    </Tooltip>
                    {/* Overflow: View Conversations, Duplicate, Export */}
                    <Dropdown
                      menu={{
                        items: [
                          {
                            key: 'quick-chat',
                            icon: <MessageCircle className="w-4 h-4" />,
                            label: t("settings:manageCharacters.actions.quickChat", {
                              defaultValue: "Test in popup"
                            }),
                            onClick: () => openQuickChat(record)
                          },
                          {
                            key: 'chat-new-tab',
                            icon: <ExternalLink className="w-4 h-4" />,
                            label: t("settings:manageCharacters.actions.chatInNewTab", {
                              defaultValue: "Chat in new tab"
                            }),
                            onClick: () => {
                              void handleChatInNewTab(record)
                            }
                          },
                          {
                            key: 'conversations',
                            icon: <History className="w-4 h-4" />,
                            label: t("settings:manageCharacters.actions.viewConversations", {
                              defaultValue: "View conversations"
                            }),
                            onClick: () => {
                              setConversationCharacter(record)
                              setCharacterChats([])
                              setChatsError(null)
                              setConversationsOpen(true)
                            }
                          },
                          {
                            key: "create-persona",
                            icon: <UserCircle2 className="w-4 h-4" />,
                            label: getCreatePersonaActionLabel(record),
                            disabled: isPersonaCreatePending(record),
                            onClick: () => {
                              void createPersonaFromCharacter(record)
                            }
                          },
                          {
                            key: "open-persona-garden",
                            icon: <ExternalLink className="w-4 h-4" />,
                            label: t(
                              "settings:manageCharacters.actions.openInPersonaGarden",
                              {
                                defaultValue: "Open in Persona Garden"
                              }
                            ),
                            onClick: () => {
                              void openPersonaGardenForCharacter(record)
                            }
                          },
                          {
                            key: "version-history",
                            icon: <Clock3 className="w-4 h-4" />,
                            label: t(
                              "settings:manageCharacters.actions.versionHistory",
                              {
                                defaultValue: "Version history"
                              }
                            ),
                            onClick: () => openVersionHistory(record)
                          },
                          {
                            key: 'duplicate',
                            icon: <Copy className="w-4 h-4" />,
                            label: duplicateLabel,
                            onClick: () => handleDuplicate(record)
                          },
                          {
                            key: isDefaultCharacter ? "clear-default" : "set-default",
                            icon: isDefaultCharacter ? (
                              <Star className="w-4 h-4" />
                            ) : (
                              <Star className="w-4 h-4" />
                            ),
                            label: isDefaultCharacter
                              ? t("settings:manageCharacters.actions.clearDefault", {
                                  defaultValue: "Clear default"
                                })
                              : t("settings:manageCharacters.actions.setDefault", {
                                  defaultValue: "Set as default"
                                }),
                            onClick: () => {
                              if (isDefaultCharacter) {
                                void handleClearDefaultCharacter()
                                return
                              }
                              void handleSetDefaultCharacter(record)
                            }
                          },
                          { type: 'divider' as const },
                          {
                            key: 'export-json',
                            icon: <Download className="w-4 h-4" />,
                            label: t("settings:manageCharacters.export.json", { defaultValue: "Export as JSON" }),
                            disabled: exporting === String(record.id || record.slug || record.name),
                            onClick: () => handleExport(record, 'json')
                          },
                          {
                            key: 'export-png',
                            icon: <Download className="w-4 h-4" />,
                            label: t("settings:manageCharacters.export.png", { defaultValue: "Export as PNG (with metadata)" }),
                            disabled: exporting === String(record.id || record.slug || record.name),
                            onClick: () => handleExport(record, 'png')
                          }
                        ]
                      }}
                      trigger={['click']}
                      placement="bottomRight">
                      <Tooltip title={t("settings:manageCharacters.actions.more", { defaultValue: "More actions" })}>
                        <button
                          type="button"
                          className="inline-flex items-center rounded-md border border-transparent p-1.5 text-text-muted transition motion-reduce:transition-none hover:border-border hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-1 focus:ring-offset-bg"
                          aria-label={t("settings:manageCharacters.aria.moreActions", {
                            defaultValue: "More actions for {{name}}",
                            name
                          })}>
                          <MoreHorizontal className="w-4 h-4" />
                        </button>
                      </Tooltip>
                    </Dropdown>
                  </div>
                )
              }
            }
          ].filter(Boolean) as any}
          />
          </div>
        </div>
      )}

      {/* Gallery View */}
      {status === "success" && Array.isArray(data) && data.length > 0 && viewMode === 'gallery' && (
        <React.Suspense fallback={null}>
          <div className="space-y-4" data-testid="characters-gallery-view">
            <div
              className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 ${
                galleryDensity === "compact" ? "gap-3" : "gap-4"
              }`}>
              {pagedGalleryData.map((character: any) => {
                const charId = String(character.id || character.slug || character.name)
                return (
                  <LazyCharacterGalleryCard
                    key={charId}
                    character={{
                      ...character,
                      tags: getCharacterVisibleTags(character?.tags)
                    }}
                    onClick={() => setPreviewCharacter(character)}
                    conversationCount={conversationCounts?.[charId]}
                    isFavorite={isCharacterFavoriteRecord(character)}
                    onToggleFavorite={() => {
                      void handleToggleFavorite(character)
                    }}
                    density={galleryDensity}
                  />
                )
              })}
            </div>
            {totalCharacters > pageSize && (
              <div className="flex justify-end">
                <Pagination
                  current={currentPage}
                  pageSize={pageSize}
                  total={totalCharacters}
                  onChange={(page, nextPageSize) => {
                    const normalizedNextPageSize = normalizePageSize(nextPageSize)
                    if (normalizedNextPageSize !== pageSize) {
                      setPageSize(normalizedNextPageSize)
                      setCurrentPage(1)
                      return
                    }
                    setCurrentPage(page)
                  }}
                  onShowSizeChange={(_page, nextPageSize) => {
                    setPageSize(normalizePageSize(nextPageSize))
                    setCurrentPage(1)
                  }}
                  showSizeChanger
                  pageSizeOptions={PAGE_SIZE_OPTIONS.map((size) => String(size))}
                />
              </div>
            )}
          </div>
        </React.Suspense>
      )}

      {/* Character Preview Popup for Gallery View */}
      {previewCharacter ? (
        <React.Suspense fallback={null}>
          <LazyCharacterPreviewPopup
            character={{
              ...previewCharacter,
              tags: getCharacterVisibleTags(previewCharacter?.tags)
            }}
            open
            onClose={() => setPreviewCharacter(null)}
            onChat={() => {
              handleChat(previewCharacter)
              setPreviewCharacter(null)
            }}
            onChatInNewTab={() => {
              void handleChatInNewTab(previewCharacter)
              setPreviewCharacter(null)
            }}
            onQuickChat={() => {
              openQuickChat(previewCharacter)
              setPreviewCharacter(null)
            }}
            onEdit={() => {
              void preloadCharacterEditor()
              handleEdit(previewCharacter)
              setPreviewCharacter(null)
            }}
            onDuplicate={() => {
              handleDuplicate(previewCharacter)
              setPreviewCharacter(null)
            }}
            onExport={async (format?: 'json' | 'png') => {
              await handleExport(previewCharacter, format || 'json')
            }}
            onDelete={async () => {
              await handleDelete(previewCharacter)
              setPreviewCharacter(null)
            }}
            onViewConversations={() => {
              handleViewConversations(previewCharacter)
              setPreviewCharacter(null)
            }}
            onCreatePersonaFromCharacter={() => {
              void createPersonaFromCharacter(previewCharacter)
              setPreviewCharacter(null)
            }}
            onOpenPersonaGarden={() => {
              void openPersonaGardenForCharacter(previewCharacter)
              setPreviewCharacter(null)
            }}
            onViewVersionHistory={() => {
              openVersionHistory(previewCharacter)
              setPreviewCharacter(null)
            }}
            creatingPersonaFromCharacter={isPersonaCreatePending(previewCharacter)}
            attachedWorldBooks={previewCharacterWorldBooks}
            attachedWorldBooksLoading={previewCharacterWorldBooksLoading}
            launchedFromWorldBooks={crossNavigationContext.launchedFromWorldBooks}
            launchedFromWorldBookId={normalizeWorldBookId(
              crossNavigationContext.focusWorldBookId
            )}
            deleting={deleting}
            exporting={
              !!exporting &&
              exporting === String(previewCharacter.id || previewCharacter.slug || previewCharacter.name)
            }
          />
        </React.Suspense>
      ) : null}
    </>
  )
}
