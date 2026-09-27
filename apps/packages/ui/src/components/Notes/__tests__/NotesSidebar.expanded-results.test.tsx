import React from 'react'
import userEvent from '@testing-library/user-event'
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within
} from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import NotesSidebar, { type NotesSidebarProps } from '../NotesSidebar'

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === 'string')
        return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return String(defaultValueOrOptions.defaultValue).replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(defaultValueOrOptions[token] ?? '')
        )
      }
      return key
    }
  })
}))

const noopAsync = vi.fn(async () => {})
const noop = vi.fn()

const baseProps = {
  isMobileViewport: false,
  mobileSidebarOpen: false,
  sidebarCollapsed: false,
  sidebarHeight: 720,
  listMode: 'active',
  listViewMode: 'list',
  page: 1,
  pageSize: 20,
  total: 1,
  sortOption: 'modified_desc',
  selectedId: null,
  visibleNotes: [],
  filteredCount: 0,
  timelineSections: [],
  recentNotes: [],
  pinnedNoteIds: [],
  pinnedNoteIdSet: new Set<string>(),
  queryInput: 'server failure',
  hasActiveFilters: true,
  activeFilterSummary: {
    countText: 'Showing 0 of 1 notes',
    detailsText: 'Query: "server failure"'
  },
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
  query: 'server failure',
  isFetching: false,
  hasListError: true,
  listErrorMessage: 'Backend unavailable',
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
  assignKeywordsToSelectedBulk: noopAsync,
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
  onOpenHealth: noop
} satisfies NotesSidebarProps

const notes = Array.from({ length: 5 }, (_, index) => ({
  id: String(index + 1),
  title: 'Note ' + (index + 1),
  content: 'Saved body',
  version: 1
}))
const expandedProps = {
  ...baseProps,
  sidebarHeight: 600,
  total: 5,
  visibleNotes: notes,
  filteredCount: 5,
  recentNotes: notes.slice(0, 3).map(({ id, title }) => ({ id, title })),
  queryInput: '',
  query: '',
  hasActiveFilters: false,
  activeFilterSummary: null,
  hasListError: false,
  listErrorMessage: null
}

describe('Notes expanded sidebar result access', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      value: 1280
    })
    Object.defineProperty(window, 'innerHeight', {
      configurable: true,
      value: 720
    })
  })
  afterEach(cleanup)
  it('bounds expanded controls and reserves a scrollable result area with five notes and three recents', async () => {
    const user = userEvent.setup()
    render(<NotesSidebar {...expandedProps} />)
    const filters = screen.getByTestId('notes-section-filters-toggle')
    if (filters.getAttribute('aria-expanded') !== 'true')
      await user.click(filters)
    expect(
      screen.getByTestId('notes-section-view-organize-toggle')
    ).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getAllByTestId(/^notes-recent-item-/)).toHaveLength(3)
    const controls = screen.getByTestId('notes-sidebar-controls')
    expect(controls.style.maxHeight).toBe('50%')
    expect(controls.className).toContain('overflow-y-auto')
    const results = screen.getByTestId('notes-sidebar-results')
    expect(results.className).toContain('min-h-[240px]')
    expect(screen.getByTestId('notes-results-scroll').className).toContain(
      'overflow-auto'
    )
    await user.click(within(results).getByTestId('notes-open-button-5'))
    expect(noopAsync).toHaveBeenCalledWith('5')
    const first = within(results).getByTestId('notes-open-button-1')
    for (let step = 0; step < 80 && document.activeElement !== first; step += 1) {
      await user.tab()
    }
    expect(first).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(noopAsync).toHaveBeenCalledWith('1')
  })
  it('retains expanded controls after reload and keeps results available on resize and mobile', async () => {
    const user = userEvent.setup()
    const view = render(<NotesSidebar {...expandedProps} />)
    const filters = screen.getByTestId('notes-section-filters-toggle')
    if (filters.getAttribute('aria-expanded') !== 'true')
      await user.click(filters)
    view.unmount()
    const restored = render(<NotesSidebar {...expandedProps} />)
    expect(screen.getByTestId('notes-section-filters-toggle')).toHaveAttribute(
      'aria-expanded',
      'true'
    )
    restored.rerender(<NotesSidebar {...expandedProps} sidebarHeight={760} />)
    fireEvent(window, new Event('resize'))
    expect(screen.getByTestId('notes-list-region')).toHaveStyle({
      height: '760px'
    })
    restored.rerender(
      <NotesSidebar {...expandedProps} isMobileViewport mobileSidebarOpen />
    )
    expect(screen.getByTestId('notes-list-region')).toHaveStyle({
      height: '100%'
    })
    await user.click(screen.getByTestId('notes-open-button-5'))
    expect(noopAsync).toHaveBeenCalledWith('5')
  })
})
