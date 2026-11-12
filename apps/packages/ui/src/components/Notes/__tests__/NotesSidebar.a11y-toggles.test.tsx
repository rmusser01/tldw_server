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

// AX-15 (#3124): the selected list mode and view must be exposed to assistive
// technology, not only shown by button fill colour.
// AX-16 (#3124): the section help icons need at least a 24x24 px target.
// AX-08 (#3124): and a focus ring in the focus colour.

describe("NotesSidebar toggle and help-button accessibility", () => {
  it("exposes the selected list mode with aria-pressed", () => {
    const { rerender } = render(<NotesSidebar {...baseProps} />)

    expect(screen.getByTestId("notes-mode-active")).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "false")

    rerender(<NotesSidebar {...baseProps} listMode="trash" />)

    expect(screen.getByTestId("notes-mode-active")).toHaveAttribute("aria-pressed", "false")
    expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "true")
  })

  it("exposes the selected view with aria-pressed", () => {
    render(<NotesSidebar {...baseProps} listViewMode="timeline" />)

    const pressed = ["list", "timeline", "inbox", "moodboard", "graph"].map((mode) => [
      mode,
      screen.getByTestId(`notes-view-mode-${mode}`).getAttribute("aria-pressed")
    ])
    expect(pressed).toEqual([
      ["list", "false"],
      ["timeline", "true"],
      ["inbox", "false"],
      ["moodboard", "false"],
      ["graph", "false"]
    ])
  })

  // XP-15 (#3124): the hint rendered at 14px in the body colour.
  it("renders the paging hint as small muted text, not antd Typography", () => {
    render(<NotesSidebar {...baseProps} showLargeListPaginationHint />)

    const hint = screen.getByTestId("notes-large-list-pagination-hint")
    expect(hint).toHaveClass("text-[11px]", "text-text-muted")
    expect(hint.className).not.toMatch(/ant-typography/)
  })

  it("gives section help icons a 24px target and a focus ring", () => {
    render(<NotesSidebar {...baseProps} />)

    const help = screen.getByRole("button", { name: "Organize help" })
    expect(help).toHaveClass("min-h-6", "min-w-6")
    expect(help).toHaveClass("focus-visible:ring-focus")
  })
})
