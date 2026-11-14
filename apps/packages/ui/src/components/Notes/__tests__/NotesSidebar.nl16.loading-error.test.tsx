import React from "react"
import { render, screen, within } from "@testing-library/react"
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

// NL-16: loading shows "Loading…" instead of "0 total"; load errors keep the
// request path behind a Details disclosure.

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
  hasListError: true,
  listErrorMessage: "Backend unavailable",
  isStaleResults: false,
  isOnline: true,
  demoEnabled: false,
  capsLoading: false,
  capabilities: { hasNotes: true } as any,
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
} satisfies NotesSidebarProps

const header = () => screen.getByTestId("notes-header-count")

describe("NotesSidebar NL-16 loading and error states", () => {
  it("says the list is loading instead of showing 0 total before the first rows arrive", () => {
    render(
      <NotesSidebar
        {...baseProps}
        total={0}
        isFetching
        hasListError={false}
        listErrorMessage={null}
        hasActiveFilters={false}
        activeFilterSummary={null}
        queryInput=""
        query=""
      />
    )

    expect(header()).toHaveTextContent("Loading…")
    expect(header()).not.toHaveTextContent("0 total")
  })

  it("shows the count once rows are in", () => {
    render(
      <NotesSidebar
        {...baseProps}
        total={3}
        filteredCount={3}
        isFetching={false}
        hasListError={false}
        listErrorMessage={null}
        hasActiveFilters={false}
        activeFilterSummary={null}
        queryInput=""
        query=""
      />
    )

    expect(header()).toHaveTextContent("3 total")
  })

  it("keeps the request path of a timeline load error behind Details", () => {
    render(
      <NotesSidebar
        {...baseProps}
        listViewMode="timeline"
        listErrorMessage="Failed to fetch (GET /api/v1/notes/?limit=20&offset=0)"
      />
    )

    const errorState = screen.getByTestId("notes-list-error-state")
    expect(within(errorState).getByText("Could not reach the notes server.")).toBeInTheDocument()
    const path = within(errorState).getByText(/GET \/api\/v1\/notes\//)
    const details = path.closest("details")
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute("open")
    expect(within(details as HTMLElement).getByText("Details")).toBeInTheDocument()
  })
})
