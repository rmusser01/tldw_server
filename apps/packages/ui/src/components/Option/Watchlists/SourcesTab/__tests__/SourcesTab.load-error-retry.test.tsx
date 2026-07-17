import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesTab } from "../SourcesTab"

const mocks = vi.hoisted(() => ({
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistTagsMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  getSourceSeenStatsMock: vi.fn(),
  exportOpmlMock: vi.fn(),
  checkWatchlistSourcesNowMock: vi.fn(),
  createWatchlistSourceMock: vi.fn(),
  deleteWatchlistSourceMock: vi.fn(),
  restoreWatchlistSourceMock: vi.fn(),
  updateWatchlistSourceMock: vi.fn(),
  tableColumnsRef: { current: [] as Array<Record<string, unknown>> },
  storeStateRef: { current: {} as Record<string, any> },
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, loading, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
      {...rest}
    >
      {children}
    </button>
  )

  const Search = ({ value, onChange, onSearch }: any) => (
    <input
      value={value || ""}
      onChange={(event) => onChange?.(event)}
      onKeyDown={(event) => {
        if (event.key === "Enter") onSearch?.(event.currentTarget.value)
      }}
    />
  )

  const Select = ({ value, onChange, options = [] }: any) => (
    <select
      value={value ?? ""}
      onChange={(event) => onChange?.(event.currentTarget.value || null)}
    >
      <option value="" />
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Alert = ({ title, description, action }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
      {action}
    </div>
  )

  return {
    Alert,
    Button,
    Empty: ({ description, children }: any) => (
      <div>
        <div>{description}</div>
        {children}
      </div>
    ),
    Input: { Search },
    Modal: { confirm: vi.fn() },
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Switch: ({ "aria-label": ariaLabel }: any) => (
      <button type="button" aria-label={ariaLabel}>
        switch
      </button>
    ),
    Table: ({ dataSource = [], columns = [], "aria-label": ariaLabel }: any) => {
      mocks.tableColumnsRef.current = Array.isArray(columns) ? columns : []
      return (
        <table data-testid="sources-table" role="table" aria-label={ariaLabel}>
          <tbody>
            {dataSource.map((record: any, rowIndex: number) => (
              <tr key={record.id ?? rowIndex}>
                {columns.map((column: any, columnIndex: number) => {
                  const key = String(column.key ?? column.dataIndex ?? columnIndex)
                  const value = column.dataIndex ? record[column.dataIndex] : undefined
                  const content = column.render ? column.render(value, record, rowIndex) : value
                  return <td key={key}>{content}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )
    },
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  checkWatchlistSourcesNow: (...args: any[]) => mocks.checkWatchlistSourcesNowMock(...args),
  createWatchlistSource: (...args: any[]) => mocks.createWatchlistSourceMock(...args),
  deleteWatchlistSource: (...args: any[]) => mocks.deleteWatchlistSourceMock(...args),
  restoreWatchlistSource: (...args: any[]) => mocks.restoreWatchlistSourceMock(...args),
  exportOpml: (...args: any[]) => mocks.exportOpmlMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  getSourceSeenStats: (...args: any[]) => mocks.getSourceSeenStatsMock(...args),
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  fetchWatchlistGroups: (...args: any[]) => mocks.fetchWatchlistGroupsMock(...args),
  fetchWatchlistTags: (...args: any[]) => mocks.fetchWatchlistTagsMock(...args),
  updateWatchlistSource: (...args: any[]) => mocks.updateWatchlistSourceMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../SourceFormModal", () => ({
  SourceFormModal: () => null
}))

vi.mock("../GroupsTree", () => ({
  GroupsTree: () => null
}))

vi.mock("../SourcesBulkImport", () => ({
  SourcesBulkImport: () => null
}))

vi.mock("../SourceSeenDrawer", () => ({
  SourceSeenDrawer: () => null
}))

const baseState = (overrides: Record<string, unknown> = {}) => ({
  sources: [],
  sourcesLoading: false,
  sourcesTotal: 0,
  sourcesSearch: "",
  sourcesPage: 1,
  sourcesPageSize: 20,
  tags: [],
  groups: [],
  groupsLoading: false,
  selectedGroupId: null,
  selectedTagName: null,
  sourceFormOpen: false,
  sourceFormEditId: null,
  setSources: vi.fn(),
  setSourcesLoading: vi.fn(),
  setSourcesSearch: vi.fn(),
  setSourcesPage: vi.fn(),
  setSourcesPageSize: vi.fn(),
  setTags: vi.fn(),
  setGroups: vi.fn(),
  setGroupsLoading: vi.fn(),
  setActiveTab: vi.fn(),
  setSelectedGroupId: vi.fn(),
  setSelectedTagName: vi.fn(),
  openSourceForm: vi.fn(),
  closeSourceForm: vi.fn(),
  addSource: vi.fn(),
  updateSourceInList: vi.fn(),
  removeSource: vi.fn(),
  ...overrides
})

describe("SourcesTab load-error retry", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.tableColumnsRef.current = []
    mocks.fetchWatchlistTagsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.getSourceSeenStatsMock.mockResolvedValue({})
    mocks.exportOpmlMock.mockResolvedValue("<opml></opml>")
    mocks.checkWatchlistSourcesNowMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistSourceMock.mockResolvedValue({})
    mocks.deleteWatchlistSourceMock.mockResolvedValue({})
    mocks.restoreWatchlistSourceMock.mockResolvedValue({})
    mocks.updateWatchlistSourceMock.mockResolvedValue({})
  })

  it("shows contextual source-load error and retries", async () => {
    mocks.fetchWatchlistSourcesMock
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce({ items: [], total: 0, has_more: false })

    render(<SourcesTab />)

    await waitFor(() => {
      expect(screen.getByText("Could not load Feeds.")).toBeInTheDocument()
      expect(screen.getByText("Check server connection and try again. Details: Failed to fetch")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(mocks.fetchWatchlistSourcesMock).toHaveBeenCalledTimes(2)
    })
  })

  it("labels the feeds table and exposes explicit active-state text", async () => {
    const sourceRecord = {
      id: 44,
      name: "AI Feed",
      url: "https://example.com/feed.xml",
      source_type: "rss",
      active: true,
      tags: [],
      group_ids: [],
      created_at: "2026-02-18T00:00:00Z",
      last_scraped_at: null,
      status: "healthy"
    }
    mocks.storeStateRef.current = baseState({
      sources: [sourceRecord],
      sourcesTotal: 1
    })

    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [sourceRecord],
      total: 1,
      has_more: false
    })

    render(<SourcesTab />)

    await waitFor(() => {
      expect(screen.getByRole("table", { name: "Feeds table" })).toBeInTheDocument()
      expect(screen.getByText("Enabled")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Toggle active: AI Feed" })).toBeInTheDocument()
    })
  })
})
