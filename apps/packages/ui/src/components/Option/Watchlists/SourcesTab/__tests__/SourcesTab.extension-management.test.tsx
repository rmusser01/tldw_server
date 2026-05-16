// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SourcesTab } from "../SourcesTab"

type SourceRecord = {
  id: number
  name: string
  url: string
  source_type: "rss" | "site" | "forum"
  active: boolean
  tags: string[]
  status: string
  group_ids?: number[]
  created_at: string
  updated_at: string
  last_scraped_at?: string | null
}

const mocks = vi.hoisted(() => ({
  checkWatchlistSourcesNowMock: vi.fn(),
  createWatchlistSourceMock: vi.fn(),
  deleteWatchlistSourceMock: vi.fn(),
  exportOpmlMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistTagsMock: vi.fn(),
  getSourceSeenStatsMock: vi.fn(),
  restoreWatchlistSourceMock: vi.fn(),
  updateWatchlistSourceMock: vi.fn(),
  modalConfirmMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> },
  tMock: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
}))

const setViewport = (width: number) => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn((query: string) => ({
      matches: query.includes("max-width") ? width <= 767 : width >= 768,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.tMock
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, icon, onClick, loading, disabled, danger: _danger, ...rest }: any) => (
    <button
      type="button"
      disabled={Boolean(loading || disabled)}
      onClick={(event) => onClick?.(event)}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
  const Search = ({ value, onChange, onSearch, placeholder, allowClear: _allowClear, ...rest }: any) => (
    <input
      aria-label={placeholder}
      placeholder={placeholder}
      value={value || ""}
      onChange={(event) => onChange?.(event)}
      onKeyDown={(event) => {
        if (event.key === "Enter") onSearch?.(event.currentTarget.value)
      }}
      {...rest}
    />
  )
  const Select = ({ value, onChange, options = [], placeholder, allowClear: _allowClear, ...rest }: any) => (
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
  const Switch = ({ checked, onChange, "aria-label": ariaLabel }: any) => (
    <button type="button" aria-label={ariaLabel} aria-pressed={Boolean(checked)} onClick={() => onChange?.(!checked)} />
  )
  const Table = ({ "aria-label": ariaLabel, dataSource = [], columns = [], rowSelection }: any) => (
    <table role="table" aria-label={ariaLabel}>
      <tbody>
        {dataSource.map((record: SourceRecord, rowIndex: number) => (
          <tr key={record.id}>
            {rowSelection ? (
              <td>
                <button
                  type="button"
                  aria-label={`Select row ${record.id}`}
                  onClick={() => rowSelection.onChange?.([record.id], [record])}
                />
              </td>
            ) : null}
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex as keyof SourceRecord] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={String(column.key ?? columnIndex)}>{content as React.ReactNode}</td>
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
    Modal: { confirm: (...args: unknown[]) => mocks.modalConfirmMock(...args) },
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Switch,
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
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  checkWatchlistSourcesNow: (...args: unknown[]) => mocks.checkWatchlistSourcesNowMock(...args),
  createWatchlistSource: (...args: unknown[]) => mocks.createWatchlistSourceMock(...args),
  deleteWatchlistSource: (...args: unknown[]) => mocks.deleteWatchlistSourceMock(...args),
  exportOpml: (...args: unknown[]) => mocks.exportOpmlMock(...args),
  fetchWatchlistGroups: (...args: unknown[]) => mocks.fetchWatchlistGroupsMock(...args),
  fetchWatchlistJobs: (...args: unknown[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistSources: (...args: unknown[]) => mocks.fetchWatchlistSourcesMock(...args),
  fetchWatchlistTags: (...args: unknown[]) => mocks.fetchWatchlistTagsMock(...args),
  getSourceSeenStats: (...args: unknown[]) => mocks.getSourceSeenStatsMock(...args),
  restoreWatchlistSource: (...args: unknown[]) => mocks.restoreWatchlistSourceMock(...args),
  updateWatchlistSource: (...args: unknown[]) => mocks.updateWatchlistSourceMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("../SourceFormModal", () => ({
  SourceFormModal: () => null
}))

vi.mock("../GroupsTree", () => ({
  GroupsTree: ({ groups = [], onSelect }: any) => (
    <div aria-label="Feed groups">
      {groups.map((group: any) => (
        <button key={group.id} type="button" onClick={() => onSelect?.(group.id)}>
          Group filter {group.name}
        </button>
      ))}
    </div>
  )
}))

vi.mock("../SourcesBulkImport", () => ({
  SourcesBulkImport: () => null
}))

vi.mock("../SourceSeenDrawer", () => ({
  SourceSeenDrawer: () => null
}))

const buildSource = (overrides: Partial<SourceRecord> = {}): SourceRecord => ({
  id: 101,
  name: "CISA KEV RSS",
  url: "https://www.cisa.gov/known-exploited-vulnerabilities-catalog.xml",
  source_type: "rss",
  active: true,
  tags: ["cve", "advisory"],
  status: "healthy",
  group_ids: [7],
  last_scraped_at: "2026-05-15T15:00:00Z",
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z",
  ...overrides
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  sources: [buildSource()],
  sourcesLoading: false,
  sourcesTotal: 1,
  sourcesSearch: "",
  sourcesPage: 1,
  sourcesPageSize: 20,
  tags: [{ id: 1, name: "cve" }],
  groups: [{ id: 7, name: "Advisories", parent_group_id: null }],
  groupsLoading: false,
  selectedGroupId: null,
  selectedTagName: null,
  sourceFormOpen: false,
  sourceFormEditId: null,
  selectedWatchlistId: 42,
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

describe("SourcesTab constrained management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState()
    mocks.checkWatchlistSourcesNowMock.mockResolvedValue({ success: 1, failed: 0, items: [] })
    mocks.createWatchlistSourceMock.mockResolvedValue(buildSource())
    mocks.deleteWatchlistSourceMock.mockResolvedValue({})
    mocks.exportOpmlMock.mockResolvedValue("<opml></opml>")
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({ items: [buildSource()], total: 1, has_more: false })
    mocks.fetchWatchlistTagsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.getSourceSeenStatsMock.mockResolvedValue({
      source_id: 101,
      defer_until: null,
      consec_not_modified: 0
    })
    mocks.restoreWatchlistSourceMock.mockResolvedValue({})
    mocks.updateWatchlistSourceMock.mockResolvedValue(buildSource())
  })

  afterEach(() => {
    cleanup()
  })

  it("replaces the feeds table with full source management cards at extension width", async () => {
    setViewport(420)
    render(<SourcesTab />)

    expect(await screen.findByTestId("watchlists-sources-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table", { name: "Feeds table" })).not.toBeInTheDocument()
    expect(screen.getByLabelText("Search sources...")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter by tag")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter by type")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Group filter Advisories" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show advanced details" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Import OPML" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add Source" })).toBeInTheDocument()

    const card = screen.getByTestId("watchlists-source-card-101")
    expect(within(card).getByText("CISA KEV RSS")).toBeInTheDocument()
    expect(within(card).getByText("https://www.cisa.gov/known-exploited-vulnerabilities-catalog.xml")).toBeInTheDocument()
    expect(within(card).getByText("RSS")).toBeInTheDocument()
    expect(within(card).getByText("cve")).toBeInTheDocument()
    expect(within(card).getByText("1 group")).toBeInTheDocument()
    expect(within(card).getByText("Healthy")).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Toggle active for CISA KEV RSS" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Check now for CISA KEV RSS" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Source Health & Dedup Stats for CISA KEV RSS" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Edit CISA KEV RSS" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Delete CISA KEV RSS" })).toBeInTheDocument()
  })

  it("keeps constrained source selection and bulk actions reachable without a table", async () => {
    setViewport(420)
    render(<SourcesTab />)

    fireEvent.click(await screen.findByRole("checkbox", { name: "Select CISA KEV RSS" }))

    expect(screen.getByText("1 selected")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Check Now" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Enable" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Disable" })).toBeInTheDocument()
    expect(screen.getByLabelText("Move to group")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Move" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Clear" })).toBeInTheDocument()
    expect(screen.queryByRole("table", { name: "Feeds table" })).not.toBeInTheDocument()
  })

  it("preserves the desktop feeds table above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<SourcesTab />)

    await waitFor(() => {
      expect(screen.getByRole("table", { name: "Feeds table" })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-sources-constrained-list")).not.toBeInTheDocument()
  })
})
