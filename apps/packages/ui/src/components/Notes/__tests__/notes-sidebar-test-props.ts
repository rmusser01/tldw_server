import { vi } from 'vitest'
import type { NotesSidebarProps } from '../NotesSidebar'

/**
 * Builds a complete, quiet NotesSidebar prop set for focused sidebar tests:
 * active list view, no notes, no filters and no list error. Every callback is
 * a fresh vi.fn so a test can assert on the one it cares about.
 */
export const buildNotesSidebarProps = (
  overrides: Partial<NotesSidebarProps> = {}
): NotesSidebarProps => {
  const noop = vi.fn()
  const noopAsync = vi.fn(async () => {})
  return {
    isMobileViewport: false,
    mobileSidebarOpen: false,
    sidebarCollapsed: false,
    listMode: 'active',
    listViewMode: 'list',
    page: 1,
    pageSize: 20,
    total: 0,
    sortOption: 'modified_desc',
    selectedId: null,
    visibleNotes: [],
    filteredCount: 0,
    timelineSections: [],
    recentNotes: [],
    pinnedNoteIds: [],
    pinnedNoteIdSet: new Set<string>(),
    queryInput: '',
    hasActiveFilters: false,
    activeFilterSummary: null,
    keywordTokens: [],
    keywordOptions: [],
    availableKeywords: [],
    notebookOptions: [],
    selectedNotebookId: null,
    selectedNotebook: null,
    moodboards: [],
    selectedMoodboardId: null,
    selectedMoodboard: null,
    isMoodboardsFetching: false,
    moodboardTotalPages: 1,
    moodboardCanGoPrev: false,
    moodboardCanGoNext: false,
    moodboardRangeStart: 0,
    moodboardRangeEnd: 0,
    bulkSelectedIds: [],
    searchTipsContent: null,
    query: '',
    isFetching: false,
    hasListError: false,
    listErrorMessage: null,
    isStaleResults: false,
    isOnline: true,
    demoEnabled: false,
    capsLoading: false,
    capabilities: { hasNotes: true } as NotesSidebarProps['capabilities'],
    queuedOfflineDraftCount: 0,
    showLargeListPaginationHint: false,
    conversationLabelById: {},
    importSubmitting: false,
    exportProgress: null,
    setMobileSidebarOpen: noop,
    setListViewMode: noop,
    setPage: noop,
    setPageSize: noop,
    setSortOption: noop,
    setQueryInput: noop,
    setSelectedMoodboardId: noop,
    setSelectedNotebookId: noop,
    setSearchTipsQuery: noop,
    handleNewNote: noopAsync,
    switchListMode: noop,
    handleSelectNote: noopAsync,
    handleClearFilters: noop,
    retryList: noop,
    handleKeywordFilterSearch: noop,
    handleKeywordFilterChange: noop,
    handleToggleBulkSelection: noop,
    clearSearchQueryTimeout: noop,
    setQuery: noop,
    openKeywordPicker: noop,
    createNotebookFromCurrentKeywords: noop,
    removeSelectedNotebook: noopAsync,
    createMoodboard: noopAsync,
    renameMoodboard: noopAsync,
    deleteMoodboard: noopAsync,
    clearBulkSelection: noop,
    exportSelectedBulk: noop,
    addTagsToSelectedBulk: noopAsync,
    deleteSelectedBulk: noopAsync,
    toggleNotePinned: noopAsync,
    restoreNote: noopAsync,
    exportAll: noopAsync,
    exportAllCSV: noopAsync,
    exportAllJSON: noopAsync,
    openImportPicker: noop,
    onSyncFolder: noop,
    resetEditor: noop,
    renderKeywordLabelWithFrequency: (keyword: string) => keyword,
    onOpenSettings: noop,
    onOpenHealth: noop,
    ...overrides
  }
}

/** Default-value-aware i18n mock used by the sidebar tests. */
export const translateWithDefaults = (
  key: string,
  defaultValueOrOptions?:
    | string
    | {
        defaultValue?: string
        [key: string]: unknown
      }
) => {
  if (typeof defaultValueOrOptions === 'string') return defaultValueOrOptions
  if (defaultValueOrOptions?.defaultValue) {
    return String(defaultValueOrOptions.defaultValue).replace(
      /\{\{(\w+)\}\}/g,
      (_match, token) => String(defaultValueOrOptions[token] ?? '')
    )
  }
  return key
}
