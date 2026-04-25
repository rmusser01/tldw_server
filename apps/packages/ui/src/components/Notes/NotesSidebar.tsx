import React from 'react'
import type { InputRef } from 'antd'
import { Input, Typography, Select, Button, Tooltip, Popover, Spin } from 'antd'
import {
  Plus as PlusIcon,
  Search as SearchIcon,
} from 'lucide-react'
import { QuestionCircleOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import NotesListPanel from '@/components/Notes/NotesListPanel'
import CollapsibleSection from '@/components/Notes/CollapsibleSection'
import type { NoteListItem } from '@/components/Notes/notes-manager-types'
import type {
  NotesSortOption,
  NotesListViewMode,
  MoodboardSummary,
  NotebookFilterOption,
  ExportProgressState,
} from './notes-manager-types'
import {
  NOTES_LIST_REGION_ID,
  MIN_SIDEBAR_HEIGHT,
} from './notes-manager-utils'
import type { ServerCapabilities } from '@/services/tldw/server-capabilities'
import type { NotesRecentOpenedEntry } from '@/services/settings/ui-settings'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ActiveFilterSummary {
  countText: string
  detailsText: string
}

export interface NotesSidebarProps {
  // Layout
  isMobileViewport: boolean
  mobileSidebarOpen: boolean
  sidebarCollapsed: boolean
  sidebarHeight: number

  // List state
  listMode: 'active' | 'trash'
  listViewMode: NotesListViewMode
  page: number
  pageSize: number
  total: number
  sortOption: NotesSortOption
  selectedId: string | number | null

  // Data
  visibleNotes: NoteListItem[]
  filteredCount: number
  timelineSections: Array<{
    key: string
    label: string
    notes: NoteListItem[]
  }>
  recentNotes: NotesRecentOpenedEntry[]
  pinnedNoteIds: string[]
  pinnedNoteIdSet: Set<string>

  // Filters
  queryInput: string
  hasActiveFilters: boolean
  activeFilterSummary: ActiveFilterSummary | null
  keywordTokens: string[]
  keywordOptions: string[]
  availableKeywords: string[]

  // Notebook
  notebookOptions: NotebookFilterOption[]
  selectedNotebookId: number | null
  selectedNotebook: NotebookFilterOption | null

  // Moodboard
  moodboards: MoodboardSummary[]
  selectedMoodboardId: number | null
  selectedMoodboard: MoodboardSummary | null
  isMoodboardsFetching: boolean
  moodboardTotalPages: number
  moodboardCanGoPrev: boolean
  moodboardCanGoNext: boolean
  moodboardRangeStart: number
  moodboardRangeEnd: number

  // Bulk selection
  bulkSelectedIds: string[]

  // Search tips
  searchTipsContent: React.ReactNode

  // Search query (committed, for NotesListPanel)
  query: string

  // Fetching / online
  isFetching: boolean
  isOnline: boolean
  demoEnabled: boolean
  capsLoading: boolean
  capabilities: ServerCapabilities | null

  // Offline drafts
  queuedOfflineDraftCount: number

  // Pagination hint
  showLargeListPaginationHint: boolean

  // NotesListPanel props
  conversationLabelById: Record<string, string>
  importSubmitting: boolean
  exportProgress: ExportProgressState | null

  // Callbacks - sidebar controls
  setMobileSidebarOpen: (open: boolean) => void
  setListViewMode: (mode: NotesListViewMode) => void
  setPage: React.Dispatch<React.SetStateAction<number>>
  setPageSize: React.Dispatch<React.SetStateAction<number>>
  setSortOption: (option: NotesSortOption) => void
  setQueryInput: (value: string) => void
  searchInputRef?: React.Ref<InputRef>
  setSelectedMoodboardId: (id: number | null) => void
  setSelectedNotebookId: (id: number | null) => void
  setSearchTipsQuery: (query: string) => void

  // Callbacks - actions
  handleNewNote: () => Promise<void>
  switchListMode: (mode: 'active' | 'trash') => void
  handleSelectNote: (id: string | number) => Promise<void>
  handleClearFilters: () => void
  handleKeywordFilterSearch: (text: string) => void
  handleKeywordFilterChange: (vals: string[] | string) => void
  handleToggleBulkSelection: (id: string | number, checked: boolean, shiftKey: boolean) => void
  clearSearchQueryTimeout: () => void
  setQuery: (query: string) => void
  openKeywordPicker: () => void
  createNotebookFromCurrentKeywords: () => void
  removeSelectedNotebook: () => Promise<void>
  createMoodboard: () => Promise<void>
  renameMoodboard: () => Promise<void>
  deleteMoodboard: () => Promise<void>
  clearBulkSelection: () => void
  exportSelectedBulk: () => void
  assignKeywordsToSelectedBulk: () => Promise<void>
  deleteSelectedBulk: () => Promise<void>
  toggleNotePinned: (id: string | number) => Promise<void>
  restoreNote: (id: string | number, version?: number) => Promise<void>
  exportAll: () => Promise<void>
  exportAllCSV: () => Promise<void>
  exportAllJSON: () => Promise<void>
  openImportPicker: () => void
  resetEditor: () => void
  renderKeywordLabelWithFrequency: (
    keyword: string,
    options?: { includeCount?: boolean; testIdPrefix?: string }
  ) => React.ReactNode

  // Navigation
  onOpenSettings: () => void
  onOpenHealth: () => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const NotesSidebar: React.FC<NotesSidebarProps> = ({
  // Layout
  isMobileViewport,
  mobileSidebarOpen,
  sidebarCollapsed,
  sidebarHeight,

  // List state
  listMode,
  listViewMode,
  page,
  pageSize,
  total,
  sortOption,
  selectedId,

  // Data
  visibleNotes,
  filteredCount,
  timelineSections,
  recentNotes,
  pinnedNoteIds,
  pinnedNoteIdSet,

  // Filters
  queryInput,
  hasActiveFilters,
  activeFilterSummary,
  keywordTokens,
  keywordOptions,
  availableKeywords,

  // Notebook
  notebookOptions,
  selectedNotebookId,
  selectedNotebook,

  // Moodboard
  moodboards,
  selectedMoodboardId,
  selectedMoodboard,
  isMoodboardsFetching,
  moodboardTotalPages,
  moodboardCanGoPrev,
  moodboardCanGoNext,
  moodboardRangeStart,
  moodboardRangeEnd,

  // Bulk selection
  bulkSelectedIds,

  // Search tips
  searchTipsContent,

  // Search query (committed)
  query,

  // Fetching / online
  isFetching,
  isOnline,
  demoEnabled,
  capsLoading,
  capabilities,

  // Offline drafts
  queuedOfflineDraftCount,

  // Pagination hint
  showLargeListPaginationHint,

  // NotesListPanel props
  conversationLabelById,
  importSubmitting,
  exportProgress,

  // Callbacks - sidebar controls
  setMobileSidebarOpen,
  setListViewMode,
  setPage,
  setPageSize,
  setSortOption,
  setQueryInput,
  searchInputRef,
  setSelectedMoodboardId,
  setSelectedNotebookId,
  setSearchTipsQuery,

  // Callbacks - actions
  handleNewNote,
  switchListMode,
  handleSelectNote,
  handleClearFilters,
  handleKeywordFilterSearch,
  handleKeywordFilterChange,
  handleToggleBulkSelection,
  clearSearchQueryTimeout,
  setQuery,
  openKeywordPicker,
  createNotebookFromCurrentKeywords,
  removeSelectedNotebook,
  createMoodboard,
  renameMoodboard,
  deleteMoodboard,
  clearBulkSelection,
  exportSelectedBulk,
  assignKeywordsToSelectedBulk,
  deleteSelectedBulk,
  toggleNotePinned,
  restoreNote,
  exportAll,
  exportAllCSV,
  exportAllJSON,
  openImportPicker,
  resetEditor,
  renderKeywordLabelWithFrequency,

  // Navigation
  onOpenSettings,
  onOpenHealth,
}) => {
  const { t } = useTranslation(['option', 'common'])
  const renderHelpButton = React.useCallback(
    (label: string) => (
      <button
        type="button"
        className="inline-flex items-center rounded-sm text-text-muted transition-colors hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        aria-label={label}
      >
        <QuestionCircleOutlined className="text-[11px]" />
      </button>
    ),
    []
  )

  // Compute active filter count for badge
  const activeFilterCount =
    keywordTokens.length +
    (selectedNotebookId != null ? 1 : 0) +
    (queryInput.trim() ? 1 : 0)

  return (
      <aside
        id={NOTES_LIST_REGION_ID}
        tabIndex={-1}
        role="region"
        aria-label={t('option:notesSearch.notesListRegionLabel', {
          defaultValue: 'Notes list'
        })}
        data-testid="notes-list-region"
        className={
          isMobileViewport
            ? `absolute left-0 top-0 z-40 h-full w-[min(92vw,420px)] max-w-full transform border-r border-border bg-surface shadow-xl transition-transform duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-focus ${
                mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'
              }`
            : `flex-shrink-0 transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-focus ${
                sidebarCollapsed ? 'w-0 overflow-hidden' : 'w-[300px] lg:w-[340px] xl:w-[380px]'
              }`
        }
        style={
          isMobileViewport
            ? { minHeight: '100%', height: '100%' }
            : { minHeight: `${MIN_SIDEBAR_HEIGHT}px`, height: `${sidebarHeight}px` }
        }
      >
        <div
          className={`flex h-full flex-col overflow-hidden border border-border bg-surface ${
            isMobileViewport ? '' : 'rounded-lg'
          }`}
        >
          {/* Toolbar Section */}
          <div className="flex-shrink-0 border-b border-border p-4 bg-surface">
            {/* ---- Always visible: Header row ---- */}
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs uppercase tracking-[0.16em] text-text-muted">
                {t('option:notesSearch.headerLabel', { defaultValue: 'Notes' })}
                <span className="ml-2 text-text-subtle">
                  {hasActiveFilters
                    ? t('option:notesSearch.headerCount', {
                        defaultValue: '{{visible}} of {{total}}',
                        visible: filteredCount,
                        total
                      })
                    : t('option:notesSearch.headerCountFallback', {
                        defaultValue: '{{total}} total',
                        total
                      })}
                </span>
              </div>
              <Tooltip
                title={t('option:notesSearch.newTooltip', {
                  defaultValue: 'Create a new note'
                })}
              >
                <Button
                  type="text"
                  shape="circle"
                  onClick={() => void handleNewNote()}
                  className="flex items-center justify-center"
                  icon={(<PlusIcon className="w-4 h-4" />)}
                  aria-label={t('option:notesSearch.new', {
                    defaultValue: 'New note'
                  })}
                />
              </Tooltip>
            </div>

            <div className="space-y-2">
              {/* ---- Collapsible: Views (always visible) ---- */}
              <CollapsibleSection
                title={t('option:notesSearch.viewOrganizeSectionTitle', {
                  defaultValue: 'Views'
                })}
                defaultOpen
                storageKey="view-organize"
                testId="notes-section-view-organize"
              >
                {showLargeListPaginationHint && (
                  <Typography.Text
                    type="secondary"
                    className="block text-[11px] text-text-muted"
                    data-testid="notes-large-list-pagination-hint"
                  >
                    {t('option:notesSearch.largeListPaginationHint', {
                      defaultValue:
                        'Showing notes in pages for faster loading.'
                    })}
                  </Typography.Text>
                )}
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    size="small"
                    type={listMode === 'active' ? 'primary' : 'default'}
                    onClick={() => {
                      void switchListMode('active')
                    }}
                    data-testid="notes-mode-active"
                  >
                    {t('option:notesSearch.modeActive', {
                      defaultValue: 'Notes'
                    })}
                  </Button>
                  <Button
                    size="small"
                    type={listMode === 'trash' ? 'primary' : 'default'}
                    onClick={() => {
                      void switchListMode('trash')
                    }}
                    data-testid="notes-mode-trash"
                  >
                    {t('option:notesSearch.modeTrash', {
                      defaultValue: 'Trash'
                    })}
                  </Button>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    size="small"
                    type={listViewMode === 'list' ? 'primary' : 'default'}
                    onClick={() => setListViewMode('list')}
                    disabled={listMode !== 'active'}
                    data-testid="notes-view-mode-list"
                  >
                    {t('option:notesSearch.viewModeList', {
                      defaultValue: 'List'
                    })}
                  </Button>
                  <Button
                    size="small"
                    type={listViewMode === 'timeline' ? 'primary' : 'default'}
                    onClick={() => setListViewMode('timeline')}
                    disabled={listMode !== 'active'}
                    data-testid="notes-view-mode-timeline"
                  >
                    {t('option:notesSearch.viewModeTimeline', {
                      defaultValue: 'Timeline'
                    })}
                  </Button>
                  <Button
                    size="small"
                    type={listViewMode === 'moodboard' ? 'primary' : 'default'}
                    onClick={() => {
                      setListViewMode('moodboard')
                      setPage(1)
                    }}
                    disabled={listMode !== 'active'}
                    data-testid="notes-view-mode-moodboard"
                  >
                    {t('option:notesSearch.viewModeMoodboard', {
                      defaultValue: 'Collection'
                    })}
                  </Button>
                </div>
                <CollapsibleSection
                  title={t('option:notesSearch.organizeSectionTitle', {
                    defaultValue: 'Organize'
                  })}
                  titleAccessory={
                    <Tooltip
                      title={t('option:notesSearch.organizeHelpTooltip', {
                        defaultValue: 'Collections group notes manually. Saved filters auto-group notes by tags.'
                      })}
                    >
                      {renderHelpButton(
                        t('option:notesSearch.organizeHelpLabel', {
                          defaultValue: 'Organize help'
                        })
                      )}
                    </Tooltip>
                  }
                  defaultOpen={false}
                  storageKey="organize"
                  testId="notes-section-organize"
                >
                {listMode === 'active' && listViewMode === 'moodboard' && (
                  <div
                    className="space-y-2 rounded border border-border bg-surface2 p-2"
                    data-testid="notes-moodboard-controls"
                  >
                    <div className="flex items-center justify-between">
                      <Typography.Text className="text-[11px] uppercase tracking-[0.08em] text-text-muted">
                        {t('option:notesSearch.moodboardLabel', {
                          defaultValue: 'Collection'
                        })}
                      </Typography.Text>
                      {isMoodboardsFetching && <Spin size="small" />}
                    </div>
                    <Select
                      className="w-full"
                      size="small"
                      value={selectedMoodboardId == null ? undefined : selectedMoodboardId}
                      onChange={(value) => {
                        if (value == null) {
                          setSelectedMoodboardId(null)
                          setPage(1)
                          return
                        }
                        const parsed = Number(value)
                        if (!Number.isFinite(parsed)) return
                        setSelectedMoodboardId(Math.floor(parsed))
                        setPage(1)
                      }}
                      placeholder={t('option:notesSearch.moodboardEmptyOption', {
                        defaultValue: 'No collections yet'
                      })}
                      options={moodboards.map((board) => ({
                        value: board.id,
                        label: board.name
                      }))}
                      data-testid="notes-moodboard-select"
                    />
                    <div className="grid grid-cols-3 gap-2">
                      <Button
                        size="small"
                        onClick={() => {
                          void createMoodboard()
                        }}
                        data-testid="notes-moodboard-create"
                      >
                        {t('option:notesSearch.moodboardCreate', {
                          defaultValue: 'New'
                        })}
                      </Button>
                      <Button
                        size="small"
                        onClick={() => {
                          void renameMoodboard()
                        }}
                        disabled={selectedMoodboard == null}
                        data-testid="notes-moodboard-rename"
                      >
                        {t('option:notesSearch.moodboardRename', {
                          defaultValue: 'Rename'
                        })}
                      </Button>
                      <Button
                        size="small"
                        danger
                        onClick={() => {
                          void deleteMoodboard()
                        }}
                        disabled={selectedMoodboard == null}
                        data-testid="notes-moodboard-delete"
                      >
                        {t('option:notesSearch.moodboardDelete', {
                          defaultValue: 'Delete'
                        })}
                      </Button>
                    </div>
                    <Typography.Text
                      type="secondary"
                      className="block text-[11px] text-text-muted"
                    >
                      {selectedMoodboard?.description
                        ? String(selectedMoodboard.description)
                        : t('option:notesSearch.moodboardHelperDefault', {
                            defaultValue:
                              'Pin notes manually or use smart rules to populate this collection.'
                          })}
                    </Typography.Text>
                  </div>
                )}
                {listMode === 'active' && (
                  <div className="space-y-1">
                    <Typography.Text
                      type="secondary"
                      className="block text-[11px] text-text-muted"
                    >
                      {t('option:notesSearch.notebookLabel', {
                        defaultValue: 'Saved filter'
                      })}
                    </Typography.Text>
                    <div className="flex items-center gap-2">
                      <Select
                        className="min-w-0 flex-1"
                        size="small"
                        value={selectedNotebookId ?? undefined}
                        onChange={(value) => {
                          if (value == null) {
                            setSelectedNotebookId(null)
                            setPage(1)
                            return
                          }
                          const parsed = Number(value)
                          if (!Number.isFinite(parsed)) {
                            setSelectedNotebookId(null)
                            return
                          }
                          setSelectedNotebookId(Math.floor(parsed))
                          setPage(1)
                        }}
                        allowClear
                        placeholder={t('option:notesSearch.notebookAllOption', {
                          defaultValue: 'All saved filters'
                        })}
                        options={notebookOptions.map((notebook) => ({
                          value: notebook.id,
                          label: `${notebook.name} (${notebook.keywords.length})`
                        }))}
                        data-testid="notes-notebook-select"
                      />
                      <Button
                        size="small"
                        onClick={createNotebookFromCurrentKeywords}
                        disabled={keywordTokens.length === 0}
                        data-testid="notes-save-notebook"
                      >
                        {t('option:notesSearch.notebookSaveAction', {
                          defaultValue: 'Save'
                        })}
                      </Button>
                      <Button
                        size="small"
                        danger
                        onClick={() => {
                          void removeSelectedNotebook()
                        }}
                        disabled={selectedNotebookId == null}
                        data-testid="notes-remove-notebook"
                      >
                        {t('option:notesSearch.notebookRemoveAction', {
                          defaultValue: 'Remove'
                        })}
                      </Button>
                    </div>
                    <Typography.Text
                      type="secondary"
                      className="block text-[11px] text-text-muted"
                      data-testid="notes-notebook-helper-text"
                    >
                      {selectedNotebook
                        ? t('option:notesSearch.notebookHelperSelected', {
                            defaultValue: 'Saved filter applies {{count}} tag filters.',
                            count: selectedNotebook.keywords.length
                          })
                        : t('option:notesSearch.notebookHelperDefault', {
                            defaultValue:
                              'Save current tag filters as saved filters.'
                          })}
                    </Typography.Text>
                  </div>
                )}
                </CollapsibleSection>
              </CollapsibleSection>

              {listMode === 'active' ? (
                <>
                  {/* ---- Always visible: Search + Sort row ---- */}
                  <div className="flex items-center gap-2">
                    <div className="min-w-0 flex-1">
                      <Input
                        allowClear
                        placeholder={t('option:notesSearch.placeholder', {
                          defaultValue: 'Search notes... (use quotes for exact match)'
                        })}
                        prefix={(<SearchIcon className="w-4 h-4 text-text-subtle" />)}
                        value={queryInput}
                        onChange={(e) => {
                          setQueryInput(e.target.value)
                        }}
                        onPressEnter={() => {
                          clearSearchQueryTimeout()
                          setQuery(queryInput)
                          setPage(1)
                        }}
                        ref={searchInputRef}
                      />
                    </div>
                    <Select
                      value={sortOption}
                      onChange={(value) => {
                        setSortOption(value as NotesSortOption)
                        setPage(1)
                      }}
                      className="w-[160px] flex-shrink-0"
                      size="small"
                      data-testid="notes-sort-select"
                      aria-label={t('option:notesSearch.sortAriaLabel', {
                        defaultValue: 'Sort notes'
                      })}
                      options={[
                        {
                          value: 'modified_desc',
                          label: t('option:notesSearch.sortModifiedDesc', {
                            defaultValue: 'Date modified (newest first)'
                          })
                        },
                        {
                          value: 'created_desc',
                          label: t('option:notesSearch.sortCreatedDesc', {
                            defaultValue: 'Date created (newest first)'
                          })
                        },
                        {
                          value: 'title_asc',
                          label: t('option:notesSearch.sortTitleAsc', {
                            defaultValue: 'Title (A-Z)'
                          })
                        },
                        {
                          value: 'title_desc',
                          label: t('option:notesSearch.sortTitleDesc', {
                            defaultValue: 'Title (Z-A)'
                          })
                        }
                      ]}
                    />
                  </div>

                  {/* ---- Collapsible: Filters ---- */}
                  <CollapsibleSection
                    title={t('option:notesSearch.filtersSectionTitle', {
                      defaultValue: 'Filters'
                    })}
                    badge={activeFilterCount > 0 ? activeFilterCount : null}
                    defaultOpen
                    storageKey="filters"
                    testId="notes-section-filters"
                  >
                    <div className="flex items-center justify-end">
                      <Popover
                        trigger="click"
                        content={searchTipsContent}
                        placement="bottomRight"
                        onOpenChange={(open) => {
                          if (!open) setSearchTipsQuery('')
                        }}
                        title={t('option:notesSearch.searchTipsTitle', {
                          defaultValue: 'Search tips'
                        })}
                      >
                        <Button
                          size="small"
                          type="link"
                          className="!px-0 text-xs"
                          data-testid="notes-search-tips-button"
                        >
                          {t('option:notesSearch.searchTipsAction', {
                            defaultValue: 'Search tips'
                          })}
                        </Button>
                      </Popover>
                    </div>
                    <Select
                      mode="tags"
                      allowClear
                      placeholder={t('option:notesSearch.keywordsPlaceholder', {
                        defaultValue: 'Filter by tag'
                      })}
                      className="w-full"
                      value={keywordTokens}
                      onSearch={handleKeywordFilterSearch}
                      onChange={handleKeywordFilterChange}
                      options={keywordOptions.map((keyword) => ({
                        label: renderKeywordLabelWithFrequency(keyword, {
                          includeCount: true,
                          testIdPrefix: 'notes-keyword-filter-option-label'
                        }),
                        value: keyword
                      }))}
                    />
                    <div className="flex items-center justify-between gap-2">
                      <div className="inline-flex items-center gap-1">
                        <Button
                          size="small"
                          onClick={openKeywordPicker}
                          disabled={!isOnline}
                          className="text-xs"
                        >
                          {t('option:notesSearch.keywordsBrowse', {
                            defaultValue: 'Browse tags'
                          })}
                        </Button>
                        <Tooltip
                          title={t('option:notesSearch.tagsHelpTooltip', {
                            defaultValue: 'Tags help you organize and filter notes. Add tags in the editor, then filter here.'
                          })}
                        >
                          {renderHelpButton(
                            t('option:notesSearch.tagsHelpLabel', {
                              defaultValue: 'Tags help'
                            })
                          )}
                        </Tooltip>
                      </div>
                      {availableKeywords.length > 0 && (
                        <Typography.Text
                          type="secondary"
                          className="text-[11px] text-text-muted"
                        >
                          {t('option:notesSearch.keywordsBrowseCount', {
                            defaultValue: '{{count}} available',
                            count: availableKeywords.length
                          })}
                        </Typography.Text>
                      )}
                    </div>
                    {activeFilterSummary && (
                      <div
                        className="rounded border border-border bg-surface2 px-2 py-1.5"
                        role="status"
                        aria-live="polite"
                        aria-label={t('option:notesSearch.activeFilterSummaryAria', {
                          defaultValue: 'Active filter summary'
                        })}
                        data-testid="notes-active-filter-summary"
                      >
                        <div className="text-[11px] font-medium text-text">
                          {activeFilterSummary.countText}
                        </div>
                        {activeFilterSummary.detailsText ? (
                          <div
                            className="mt-1 text-[11px] text-text-muted"
                            data-testid="notes-active-filter-summary-details"
                          >
                            {activeFilterSummary.detailsText}
                          </div>
                        ) : null}
                      </div>
                    )}
                    <Typography.Text
                      type="secondary"
                      className="block text-[11px] text-text-muted"
                      data-testid="notes-in-note-search-guidance"
                    >
                      {t('option:notesSearch.inNoteSearchGuidance', {
                        defaultValue: 'For in-note search, use browser Ctrl/Cmd+F.'
                      })}
                    </Typography.Text>
                    {hasActiveFilters && (
                      <Button
                        size="small"
                        onClick={handleClearFilters}
                        className="w-full text-xs"
                        aria-label={t('option:notesSearch.clearAria', {
                          defaultValue: 'Clear active note filters'
                        })}
                      >
                        {t('option:notesSearch.clear', {
                          defaultValue: 'Clear search & filters'
                        })}
                      </Button>
                    )}
                  </CollapsibleSection>

                  {/* ---- Recent notes (outside collapsibles) ---- */}
                  {recentNotes.length > 0 && (
                    <div
                      className="rounded border border-border bg-surface2 p-2"
                      data-testid="notes-recent-section"
                    >
                      <div className="text-[11px] uppercase tracking-[0.08em] text-text-muted">
                        {t('option:notesSearch.recentNotesHeading', {
                          defaultValue: 'Recent notes'
                        })}
                      </div>
                      <div className="mt-1 space-y-1">
                        {recentNotes.map((recent) => (
                          <button
                            key={recent.id}
                            type="button"
                            className="block w-full truncate rounded px-2 py-1 text-left text-xs text-text hover:bg-surface3"
                            onClick={() => {
                              void handleSelectNote(recent.id)
                            }}
                            data-testid={`notes-recent-item-${recent.id.replace(/[^a-z0-9_-]/gi, '_')}`}
                          >
                            {recent.title}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <Typography.Text
                  type="secondary"
                  className="text-[11px] text-text-muted block"
                >
                  {t('option:notesSearch.trashHelpText', {
                    defaultValue: 'Restore notes from trash to edit them again.'
                  })}
                </Typography.Text>
              )}
            </div>
          </div>

          {/* Notes List Section */}
          <div className="flex-1 overflow-y-auto">
            {listMode === 'active' && listViewMode === 'timeline' ? (
              <div className="h-full overflow-y-auto px-3 py-3" data-testid="notes-timeline-view">
                {isFetching && (
                  <div className="mb-3 inline-flex items-center gap-2 text-xs text-text-muted">
                    <Spin size="small" />
                    <span>
                      {t('option:notesSearch.timelineLoading', {
                        defaultValue: 'Loading notes...'
                      })}
                    </span>
                  </div>
                )}
                {timelineSections.length === 0 ? (
                  <div
                    className="rounded-md border border-dashed border-border bg-surface2 px-3 py-4 text-sm text-text-muted"
                    data-testid="notes-timeline-empty"
                  >
                    {t('option:notesSearch.timelineEmpty', {
                      defaultValue: 'No notes match the current filters.'
                    })}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {timelineSections.map((section) => (
                      <section key={section.key} data-testid={`notes-timeline-group-${section.key}`}>
                        <h3 className="mb-2 text-[11px] uppercase tracking-[0.08em] text-text-muted">
                          {section.label}
                        </h3>
                        <div className="space-y-2">
                          {section.notes.map((note) => {
                            const noteId = String(note.id)
                            const isSelected = selectedId != null && String(selectedId) === noteId
                            const updatedLabel = note.updated_at
                              ? new Date(note.updated_at).toLocaleString()
                              : t('option:notesSearch.timelineUnknownDate', {
                                  defaultValue: 'Unknown date'
                                })
                            return (
                              <button
                                key={noteId}
                                type="button"
                                onClick={() => {
                                  void handleSelectNote(note.id)
                                }}
                                data-testid={`notes-timeline-item-${noteId.replace(/[^a-z0-9_-]/gi, '_')}`}
                                className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
                                  isSelected
                                    ? 'border-primary bg-primary/10'
                                    : 'border-border bg-surface hover:bg-surface2'
                                }`}
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <span className="truncate text-sm font-medium text-text">
                                    {String(note.title || `Note ${noteId}`)}
                                  </span>
                                  {pinnedNoteIdSet.has(noteId) && (
                                    <span className="text-[10px] uppercase tracking-[0.08em] text-primary">
                                      {t('option:notesSearch.timelinePinned', {
                                        defaultValue: 'Pinned'
                                      })}
                                    </span>
                                  )}
                                </div>
                                <div className="mt-1 text-[11px] text-text-muted">{updatedLabel}</div>
                              </button>
                            )
                          })}
                        </div>
                      </section>
                    ))}
                  </div>
                )}
              </div>
            ) : listMode === 'active' && listViewMode === 'moodboard' ? (
              <div className="h-full overflow-y-auto px-3 py-3" data-testid="notes-moodboard-view">
                {selectedMoodboard == null ? (
                  <div
                    className="rounded-md border border-dashed border-border bg-surface2 px-3 py-4 text-sm text-text-muted"
                    data-testid="notes-moodboard-empty-selection"
                  >
                    {t('option:notesSearch.moodboardSelectPrompt', {
                      defaultValue: 'Create or select a collection to start.'
                    })}
                  </div>
                ) : isFetching ? (
                  <div className="mb-3 inline-flex items-center gap-2 text-xs text-text-muted">
                    <Spin size="small" />
                    <span>
                      {t('option:notesSearch.moodboardLoading', {
                        defaultValue: 'Loading collection...'
                      })}
                    </span>
                  </div>
                ) : visibleNotes.length === 0 ? (
                  <div
                    className="rounded-md border border-dashed border-border bg-surface2 px-3 py-4 text-sm text-text-muted"
                    data-testid="notes-moodboard-empty"
                  >
                    {t('option:notesSearch.moodboardEmpty', {
                      defaultValue: 'No notes in this collection yet.'
                    })}
                  </div>
                ) : (
                  <>
                    <div className="columns-1 gap-3 sm:columns-2 xl:columns-3">
                      {visibleNotes.map((note) => {
                        const noteId = String(note.id)
                        const isSelected = selectedId != null && String(selectedId) === noteId
                        const preview = String(note.content_preview || note.content || '').trim()
                        const membership = String(note.membership_source || 'manual')
                        const membershipLabel =
                          membership === 'both'
                            ? t('option:notesSearch.moodboardMembershipBoth', { defaultValue: 'Pinned + Rule-matched' })
                            : membership === 'smart'
                              ? t('option:notesSearch.moodboardMembershipSmart', { defaultValue: 'Rule-matched' })
                              : t('option:notesSearch.moodboardMembershipManual', { defaultValue: 'Pinned' })
                        return (
                          <button
                            key={noteId}
                            type="button"
                            data-testid={`notes-moodboard-card-${noteId.replace(/[^a-z0-9_-]/gi, '_')}`}
                            onClick={() => {
                              void handleSelectNote(note.id)
                            }}
                            className={`mb-3 w-full break-inside-avoid overflow-hidden rounded-lg border text-left transition-colors ${
                              isSelected
                                ? 'border-primary bg-primary/10'
                                : 'border-border bg-surface hover:bg-surface2'
                            }`}
                          >
                            {note.cover_image_url ? (
                              <img
                                src={note.cover_image_url}
                                alt={String(note.title || `Note ${noteId}`)}
                                className="h-32 w-full object-cover"
                              />
                            ) : (
                              <div className="flex h-24 w-full items-center justify-center bg-surface3 text-xl font-semibold text-text-muted">
                                {String(note.title || '').trim().slice(0, 1).toUpperCase() || '#'}
                              </div>
                            )}
                            <div className="space-y-2 p-3">
                              <div className="line-clamp-2 text-sm font-medium text-text">
                                {String(note.title || `Note ${noteId}`)}
                              </div>
                              {preview ? (
                                <div className="line-clamp-4 text-xs text-text-muted">{preview}</div>
                              ) : null}
                              <div className="flex flex-wrap items-center gap-1.5">
                                <span className="rounded-full bg-surface3 px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-text-muted">
                                  {membershipLabel}
                                </span>
                                {(note.keywords || []).slice(0, 2).map((keyword) => (
                                  <span
                                    key={`${noteId}-${keyword}`}
                                    className="rounded-full border border-border px-2 py-0.5 text-[10px] text-text-muted"
                                  >
                                    {keyword}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </button>
                        )
                      })}
                    </div>
                    <div
                      className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded border border-border bg-surface px-3 py-2"
                      data-testid="notes-moodboard-pagination"
                    >
                      <Typography.Text
                        type="secondary"
                        className="text-xs text-text-muted"
                        data-testid="notes-moodboard-page-summary"
                      >
                        {t('option:notesSearch.moodboardPageSummary', {
                          defaultValue: 'Showing {{start}}-{{end}} of {{total}}',
                          start: moodboardRangeStart,
                          end: moodboardRangeEnd,
                          total
                        })}
                      </Typography.Text>
                      <div className="flex items-center gap-2">
                        <Select
                          value={pageSize}
                          onChange={(value) => {
                            const parsed = Number(value)
                            if (!Number.isFinite(parsed) || parsed <= 0) return
                            setPageSize(Math.floor(parsed))
                            setPage(1)
                          }}
                          size="small"
                          data-testid="notes-moodboard-page-size"
                          aria-label={t('option:notesSearch.moodboardPageSizeAria', {
                            defaultValue: 'Collection page size'
                          })}
                          options={[10, 20, 50, 100].map((size) => ({
                            value: size,
                            label: t('option:notesSearch.pageSizeValue', {
                              defaultValue: '{{size}} / page',
                              size
                            })
                          }))}
                        />
                        <Button
                          size="small"
                          onClick={() => setPage((current) => Math.max(1, current - 1))}
                          disabled={!moodboardCanGoPrev}
                          data-testid="notes-moodboard-page-prev"
                        >
                          {t('option:notesSearch.pagePrev', {
                            defaultValue: 'Prev'
                          })}
                        </Button>
                        <Typography.Text
                          className="text-xs text-text"
                          data-testid="notes-moodboard-page-index"
                        >
                          {t('option:notesSearch.pageIndexLabel', {
                            defaultValue: 'Page {{page}} / {{pages}}',
                            page,
                            pages: moodboardTotalPages
                          })}
                        </Typography.Text>
                        <Button
                          size="small"
                          onClick={() => setPage((current) => Math.min(moodboardTotalPages, current + 1))}
                          disabled={!moodboardCanGoNext}
                          data-testid="notes-moodboard-page-next"
                        >
                          {t('option:notesSearch.pageNext', {
                            defaultValue: 'Next'
                          })}
                        </Button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <NotesListPanel
                listMode={listMode}
                searchQuery={query}
                conversationLabelById={conversationLabelById}
                bulkSelectedIds={bulkSelectedIds}
                isOnline={isOnline}
                isFetching={isFetching}
                demoEnabled={demoEnabled}
                capsLoading={capsLoading}
                capabilities={capabilities || null}
                notes={visibleNotes}
                total={total}
                page={page}
                pageSize={pageSize}
                selectedId={selectedId}
                pinnedNoteIds={pinnedNoteIds}
                onSelectNote={(id) => {
                  void handleSelectNote(id)
                }}
                onToggleBulkSelection={handleToggleBulkSelection}
                onTogglePinned={(id) => {
                  void toggleNotePinned(id)
                }}
                onChangePage={(nextPage, nextPageSize) => {
                  const normalizedPageSize = Number(nextPageSize || pageSize)
                  const sizeChanged = normalizedPageSize !== pageSize
                  setPageSize(normalizedPageSize)
                  setPage(sizeChanged ? 1 : nextPage)
                }}
                onResetEditor={() => {
                  if (listMode === 'trash') {
                    void switchListMode('active')
                    return
                  }
                  resetEditor()
                }}
                onOpenSettings={onOpenSettings}
                onOpenHealth={onOpenHealth}
                onRestoreNote={(id, version) => {
                  void restoreNote(id, version)
                }}
                onExportAllMd={() => {
                  void exportAll()
                }}
                onExportAllCsv={() => {
                  void exportAllCSV()
                }}
                onExportAllJson={() => {
                  void exportAllJSON()
                }}
                onImportNotes={openImportPicker}
                importInProgress={importSubmitting}
                exportProgress={exportProgress}
              />
            )}
          </div>

          {/* Offline draft sync indicator */}
          {queuedOfflineDraftCount > 0 && (
            <div
              className="flex-shrink-0 border-t border-border px-4 py-1.5"
              data-testid="notes-sidebar-offline-status"
            >
              <Typography.Text className="text-[11px] text-text-muted">
                {t('option:notesSearch.sidebarOfflineDrafts', {
                  defaultValue: '{{count}} draft(s) pending sync',
                  count: queuedOfflineDraftCount
                })}
              </Typography.Text>
            </div>
          )}

          {/* Sticky bulk actions bar at bottom */}
          {listMode === 'active' && bulkSelectedIds.length > 0 && (
            <div
              className="flex-shrink-0 border-t border-border bg-surface2 px-4 py-2"
              role="status"
              aria-live="polite"
              data-testid="notes-bulk-actions-bar"
            >
              <div className="flex items-center justify-between gap-2">
                <Typography.Text
                  className="text-xs font-medium text-text"
                  data-testid="notes-bulk-selected-count"
                >
                  {t('option:notesSearch.bulkSelectedCount', {
                    defaultValue: '{{count}} selected',
                    count: bulkSelectedIds.length
                  })}
                </Typography.Text>
                <Button
                  size="small"
                  type="link"
                  className="!px-0 text-xs"
                  onClick={clearBulkSelection}
                  data-testid="notes-bulk-clear-selection"
                >
                  {t('option:notesSearch.bulkClearSelection', {
                    defaultValue: 'Clear selection'
                  })}
                </Button>
              </div>
              <div className="mt-2 grid grid-cols-3 gap-2">
                <Button
                  size="small"
                  className={isMobileViewport ? 'min-h-[44px]' : undefined}
                  onClick={exportSelectedBulk}
                  data-testid="notes-bulk-export"
                >
                  {t('option:notesSearch.bulkExport', {
                    defaultValue: 'Export selected'
                  })}
                </Button>
                <Button
                  size="small"
                  className={isMobileViewport ? 'min-h-[44px]' : undefined}
                  onClick={() => {
                    void assignKeywordsToSelectedBulk()
                  }}
                  data-testid="notes-bulk-assign-keywords"
                >
                  {t('option:notesSearch.bulkAssignKeywords', {
                    defaultValue: 'Assign tags'
                  })}
                </Button>
                <Button
                  size="small"
                  className={isMobileViewport ? 'min-h-[44px]' : undefined}
                  danger
                  onClick={() => {
                    void deleteSelectedBulk()
                  }}
                  data-testid="notes-bulk-delete"
                >
                  {t('option:notesSearch.bulkDelete', {
                    defaultValue: 'Delete selected'
                  })}
                </Button>
              </div>
            </div>
          )}
        </div>
      </aside>
  )
}

export default React.memo(NotesSidebar)
