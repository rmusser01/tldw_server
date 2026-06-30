// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesTab } from "../SourcesTab"

const ADVANCED_COLUMNS_STORAGE_KEY = "watchlists:sources:advanced-columns:v1"

type SourceRecord = {
  id: number
  name: string
  url: string
  source_type: "rss" | "site" | "forum"
  active: boolean
  tags: string[]
  group_ids?: number[]
  watchlist_ids?: number[]
  settings?: Record<string, unknown> | null
  status: string
  created_at: string
  updated_at: string
  last_scraped_at: string | null
}

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
  sourceFormModalProps: { current: null as any },
  showUndoNotificationMock: vi.fn(),
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  },
  storeStateRef: { current: {} as Record<string, any> }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, loading, disabled, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading || disabled)}
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

  const Select = ({
    value,
    onChange,
    options = [],
    placeholder,
    allowClear: _allowClear,
    className: _className,
    size: _size,
    ...rest
  }: any) => (
    <select
      aria-label={placeholder ?? "select"}
      value={value ?? ""}
      onChange={(event) => onChange?.(event.currentTarget.value || null)}
      {...rest}
    >
      <option value="" />
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <table data-testid="sources-table">
      <tbody>
        {dataSource.map((record: SourceRecord, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const key = String(column.key ?? column.dataIndex ?? columnIndex)
              const value = column.dataIndex ? record[column.dataIndex as keyof SourceRecord] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={key}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )

  return {
    Alert: ({ title, description, action }: any) => (
      <div>
        <div>{title}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
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
    Switch: () => null,
    Table,
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
    showUndoNotification: (...args: any[]) => mocks.showUndoNotificationMock(...args)
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
  SourceFormModal: (props: any) => {
    mocks.sourceFormModalProps.current = props
    if (!props.open) return null
    const initialValues = props.initialValues || {}
    return (
      <div data-testid="source-form-modal">
        <div data-testid="source-form-mode">{props.mode || ""}</div>
        <div data-testid="source-form-name">{initialValues.name || ""}</div>
        <div data-testid="source-form-url">{initialValues.url || ""}</div>
        <button
          type="button"
          data-testid="source-form-submit"
          onClick={() =>
            props.onSubmit({
              name: initialValues.name,
              url: "https://copy.example.com/feed.xml",
              source_type: initialValues.source_type,
              tags: initialValues.tags,
              settings: initialValues.settings
            })
          }
        >
          submit
        </button>
      </div>
    )
  }
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

const buildSource = (id: number): SourceRecord => ({
  id,
  name: `Feed ${id}`,
  url: `https://example.com/feed-${id}.xml`,
  source_type: "rss",
  active: true,
  tags: ["tech", "ai"],
  group_ids: [7],
  watchlist_ids: [3],
  settings: {
    fetch_mode: "rss",
    dedupe_identity: ["canonical_url", "title"]
  },
  status: "healthy",
  created_at: "2026-02-20T00:00:00Z",
  updated_at: "2026-02-20T00:00:00Z",
  last_scraped_at: "2026-02-21T00:00:00Z"
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  sources: [buildSource(101)],
  sourcesLoading: false,
  sourcesTotal: 1,
  sourcesSearch: "",
  sourcesPage: 1,
  sourcesPageSize: 20,
  tags: [],
  groups: [],
  groupsLoading: false,
  selectedGroupId: null,
  selectedTagName: null,
  selectedWatchlistId: 42,
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
  openSourceForm: vi.fn((editId?: number | null) => {
    mocks.storeStateRef.current = {
      ...mocks.storeStateRef.current,
      sourceFormOpen: true,
      sourceFormEditId: editId ?? null
    }
  }),
  closeSourceForm: vi.fn(() => {
    mocks.storeStateRef.current = {
      ...mocks.storeStateRef.current,
      sourceFormOpen: false,
      sourceFormEditId: null
    }
  }),
  addSource: vi.fn(),
  updateSourceInList: vi.fn(),
  removeSource: vi.fn(),
  ...overrides
})

describe("SourcesTab advanced details disclosure", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem(ADVANCED_COLUMNS_STORAGE_KEY)

    mocks.storeStateRef.current = baseState()
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [buildSource(101)],
      total: 1,
      page: 1,
      size: 20,
      has_more: false
    })
    mocks.fetchWatchlistTagsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.getSourceSeenStatsMock.mockResolvedValue({
      source_id: 101,
      defer_until: null,
      consec_not_modified: 0
    })
    mocks.exportOpmlMock.mockResolvedValue("<opml></opml>")
    mocks.checkWatchlistSourcesNowMock.mockResolvedValue({ items: [] })
    mocks.createWatchlistSourceMock.mockResolvedValue({})
    mocks.deleteWatchlistSourceMock.mockResolvedValue({})
    mocks.restoreWatchlistSourceMock.mockResolvedValue({})
    mocks.updateWatchlistSourceMock.mockResolvedValue(buildSource(101))
  })

  afterEach(() => {
    localStorage.removeItem(ADVANCED_COLUMNS_STORAGE_KEY)
  })

  it("starts with compact summaries and expands advanced columns on demand", async () => {
    render(<SourcesTab />)

    expect(await screen.findByTestId("source-compact-summary-101")).toHaveTextContent("1 group • 2 tags")
    expect(screen.getByTestId("watchlists-sources-density-hint")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-sources-advanced-toggle"))

    await waitFor(() => {
      expect(screen.getByText("tech")).toBeInTheDocument()
      expect(screen.getByText("ai")).toBeInTheDocument()
    })
    expect(screen.queryByTestId("source-compact-summary-101")).not.toBeInTheDocument()
    expect(localStorage.getItem(ADVANCED_COLUMNS_STORAGE_KEY)).toBe("1")
  })

  it.each([1, 10, 50])(
    "keeps feed table summaries actionable for %i feeds in compact and advanced density modes",
    async (feedCount) => {
      const sources = Array.from({ length: feedCount }, (_unused, index) => buildSource(1000 + index))
      mocks.storeStateRef.current = baseState({
        sources,
        sourcesTotal: feedCount
      })
      mocks.fetchWatchlistSourcesMock.mockResolvedValue({
        items: sources,
        total: feedCount,
        page: 1,
        size: feedCount,
        has_more: false
      })

      render(<SourcesTab />)

      await waitFor(() => {
        expect(screen.getByTestId("source-compact-summary-1000")).toBeInTheDocument()
      })
      expect(document.querySelectorAll("[data-testid^='source-compact-summary-']")).toHaveLength(feedCount)

      fireEvent.click(screen.getByTestId("watchlists-sources-advanced-toggle"))

      await waitFor(() => {
        expect(screen.getAllByText("tech").length).toBeGreaterThan(0)
      })
      expect(localStorage.getItem(ADVANCED_COLUMNS_STORAGE_KEY)).toBe("1")
    }
  )

  it("reuses cached group OPML URLs across refreshes while cache is fresh", async () => {
    const sourceA = buildSource(301)
    const sourceB = {
      ...buildSource(302),
      url: "https://example.com/other-feed.xml"
    }
    const sources = [sourceA, sourceB]

    mocks.storeStateRef.current = baseState({
      selectedGroupId: 7,
      groups: [{ id: 7, name: "Priority", description: null, parent_group_id: null }],
      sources,
      sourcesTotal: 2
    })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: sources,
      total: 2,
      page: 1,
      size: 200,
      has_more: false
    })
    mocks.exportOpmlMock.mockResolvedValue(
      `<opml><body><outline text="Priority"><outline xmlUrl="${sourceA.url}" /></outline></body></opml>`
    )

    render(<SourcesTab />)

    await waitFor(() => {
      expect(mocks.exportOpmlMock).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(screen.getByText("Refresh"))

    await waitFor(() => {
      expect(mocks.fetchWatchlistSourcesMock).toHaveBeenCalledTimes(2)
    })
    expect(mocks.exportOpmlMock).toHaveBeenCalledTimes(1)
  })

  it("opens cloned feeds in the source form before creating so duplicate URLs can be edited", async () => {
    const source = {
      ...buildSource(101),
      group_ids: [7, 8],
      watchlist_ids: [11],
      settings: {
        fetch: { user_agent: "tldw-test" },
        extraction: { title_selector: "h1" },
        dedupe: { identity: ["canonical_url", "title"] }
      }
    }
    mocks.storeStateRef.current = baseState({
      sources: [source],
      sourcesTotal: 1,
      selectedWatchlistId: 42
    })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [source],
      total: 1,
      page: 1,
      size: 20,
      has_more: false
    })

    render(<SourcesTab />)

    fireEvent.click(await screen.findByRole("button", { name: "Clone Feed 101" }))

    expect(mocks.createWatchlistSourceMock).not.toHaveBeenCalled()
    expect(mocks.storeStateRef.current.openSourceForm).toHaveBeenCalledWith()
    expect(await screen.findByTestId("source-form-modal")).toBeInTheDocument()
    expect(screen.getByTestId("source-form-mode")).toHaveTextContent("create")
    expect(screen.getByTestId("source-form-name")).toHaveTextContent("Feed 101 copy")
    expect(screen.getByTestId("source-form-url")).toHaveTextContent("https://example.com/feed-101.xml")

    fireEvent.click(screen.getByTestId("source-form-submit"))

    await waitFor(() => {
      expect(mocks.createWatchlistSourceMock).toHaveBeenCalledWith({
        name: "Feed 101 copy",
        url: "https://copy.example.com/feed.xml",
        source_type: "rss",
        active: false,
        tags: ["tech", "ai"],
        settings: source.settings,
        group_ids: [7, 8],
        watchlist_id: 42
      })
    })
  })
})
