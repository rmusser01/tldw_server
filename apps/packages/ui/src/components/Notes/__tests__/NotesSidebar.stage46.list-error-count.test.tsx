import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesSidebar, { type NotesSidebarProps } from "../NotesSidebar"

vi.mock("react-i18next", () => ({
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
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return String(defaultValueOrOptions.defaultValue).replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(defaultValueOrOptions[token] ?? "")
        )
      }
      return key
    }
  })
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

const noopAsync = vi.fn(async () => {})
const noop = vi.fn()

const baseProps = {
  isMobileViewport: false,
  mobileSidebarOpen: false,
  sidebarCollapsed: false,
  sidebarHeight: 720,
  listMode: "active",
  listViewMode: "list",
  page: 1,
  pageSize: 20,
  total: 1,
  sortOption: "modified_desc",
  selectedId: null,
  visibleNotes: [],
  filteredCount: 0,
  timelineSections: [],
  recentNotes: [],
  pinnedNoteIds: [],
  pinnedNoteIdSet: new Set<string>(),
  queryInput: "server failure",
  hasActiveFilters: true,
  activeFilterSummary: {
    countText: "Showing 0 of 1 notes",
    detailsText: 'Query: "server failure"',
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
  query: "server failure",
  isFetching: false,
  isListError: true,
  listError: new Error("Backend unavailable"),
  isStaleResults: false,
  isOnline: true,
  demoEnabled: false,
  capsLoading: false,
  capabilities: { hasNotes: true },
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
  resetEditor: noop,
  renderKeywordLabelWithFrequency: (keyword: string) => keyword,
  onOpenSettings: noop,
  onOpenHealth: noop,
} satisfies NotesSidebarProps

describe("NotesSidebar list error count state", () => {
  it("does not present filtered totals as fresh when the list refresh failed", () => {
    render(<NotesSidebar {...baseProps} />)

    expect(screen.getAllByText("Refresh failed").length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText("0 of 1")).not.toBeInTheDocument()
    expect(screen.queryByText("Showing 0 of 1 notes")).not.toBeInTheDocument()
  })
})
