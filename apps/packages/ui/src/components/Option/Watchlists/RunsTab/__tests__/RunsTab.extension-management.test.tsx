// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"
import { RunDetailDrawer } from "../RunDetailDrawer"

const mocks = vi.hoisted(() => ({
  cancelWatchlistRunMock: vi.fn(),
  exportRunTalliesCsvMock: vi.fn(),
  exportRunsCsvMock: vi.fn(),
  fetchJobRunsMock: vi.fn(),
  fetchScrapedItemsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistOutputsMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  getRunDetailsMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  updateScrapedItemMock: vi.fn(),
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
  const Select = ({ value, onChange, options = [], placeholder, allowClear, ...rest }: any) => (
    <select
      aria-label={placeholder ?? "select"}
      data-testid={rest["data-testid"]}
      value={value == null ? "" : String(value)}
      onChange={(event) => onChange?.(event.currentTarget.value || null)}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )
  const Table = ({ "aria-label": ariaLabel, dataSource = [], columns = [] }: any) => (
    <table role="table" aria-label={ariaLabel || "table"}>
      <tbody>
        {dataSource.map((record: any, rowIndex: number) => (
          <tr key={record.id ?? rowIndex}>
            {columns.map((column: any, columnIndex: number) => {
              const value = column.dataIndex ? record[column.dataIndex] : undefined
              const content = column.render ? column.render(value, record, rowIndex) : value
              return <td key={String(column.key ?? column.dataIndex ?? columnIndex)}>{content}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
  const Tabs = ({ items = [] }: any) => (
    <div>
      {items.map((item: any) => (
        <section key={item.key}>
          <h2>{item.label}</h2>
          {item.children}
        </section>
      ))}
    </div>
  )
  const DescriptionsComponent = ({ children }: any) => <div>{children}</div>
  ;(DescriptionsComponent as any).Item = ({ label, children }: any) => (
    <div>
      <span>{label}</span>
      {children}
    </div>
  )
  const Switch = ({ checked, onChange, loading, "aria-label": ariaLabel }: any) => (
    <button
      type="button"
      aria-label={ariaLabel}
      aria-pressed={Boolean(checked)}
      disabled={Boolean(loading)}
      onClick={() => onChange?.(!checked)}
    />
  )
  return {
    Alert: ({ title, message, description, action }: any) => (
      <div>
        <div>{title || message}</div>
        <div>{description}</div>
        {action}
      </div>
    ),
    Button,
    Descriptions: DescriptionsComponent,
    Drawer: ({ open, title, extra, children }: any) => open ? (
      <div>
        <div>{title}</div>
        {extra}
        {children}
      </div>
    ) : null,
    Dropdown: ({ children }: any) => <>{children}</>,
    Empty: ({ description }: any) => <div>{description}</div>,
    Progress: () => <div />,
    Select,
    Space: ({ children }: any) => <>{children}</>,
    Spin: () => <div>Loading</div>,
    Switch,
    Table,
    Tabs,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children, title }: any) => (
      <>
        {children}
        {typeof title === "string" ? <span>{title}</span> : null}
      </>
    ),
    message: {
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn()
    }
  }
})

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("@/services/watchlists", () => ({
  cancelWatchlistRun: (...args: unknown[]) => mocks.cancelWatchlistRunMock(...args),
  exportRunTalliesCsv: (...args: unknown[]) => mocks.exportRunTalliesCsvMock(...args),
  exportRunsCsv: (...args: unknown[]) => mocks.exportRunsCsvMock(...args),
  fetchJobRuns: (...args: unknown[]) => mocks.fetchJobRunsMock(...args),
  fetchScrapedItems: (...args: unknown[]) => mocks.fetchScrapedItemsMock(...args),
  fetchWatchlistJobs: (...args: unknown[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistOutputs: (...args: unknown[]) => mocks.fetchWatchlistOutputsMock(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchWatchlistRunsMock(...args),
  fetchWatchlistSources: (...args: unknown[]) => mocks.fetchWatchlistSourcesMock(...args),
  getRunDetails: (...args: unknown[]) => mocks.getRunDetailsMock(...args),
  triggerWatchlistRun: (...args: unknown[]) => mocks.triggerWatchlistRunMock(...args),
  updateScrapedItem: (...args: unknown[]) => mocks.updateScrapedItemMock(...args)
}))

vi.mock("@/services/watchlists-stream", () => ({
  buildWatchlistsRunWebSocketUrl: () => "ws://example.test",
  parseWatchlistsRunStreamPayload: () => null
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(async () => null)
  }
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

const buildRun = (overrides: Record<string, unknown> = {}) => ({
  id: 55,
  job_id: 8,
  status: "running",
  started_at: "2026-05-15T12:00:00Z",
  finished_at: null,
  stats: {
    items_found: 12,
    items_ingested: 8,
    items_filtered: 3,
    items_errored: 1
  },
  error_msg: null,
  log_path: null,
  ...overrides
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  activeTab: "runs",
  runs: [buildRun()],
  runsLoading: false,
  runsTotal: 1,
  runsPage: 1,
  runsPageSize: 20,
  runsJobFilter: null,
  runsStatusFilter: null,
  pollingActive: false,
  runDetailOpen: false,
  selectedRunId: null,
  selectedWatchlistId: 42,
  setRuns: vi.fn(),
  setRunsLoading: vi.fn(),
  setRunsPage: vi.fn(),
  setRunsPageSize: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setOutputsJobFilter: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  setActiveTab: vi.fn(),
  setPollingActive: vi.fn(),
  openRunDetail: vi.fn(),
  closeRunDetail: vi.fn(),
  updateRunInList: vi.fn(),
  addRun: vi.fn(),
  openJobForm: vi.fn(),
  ...overrides
})

describe("RunsTab constrained management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mocks.storeStateRef.current = baseState()
    mocks.cancelWatchlistRunMock.mockResolvedValue({ cancelled: true })
    mocks.exportRunTalliesCsvMock.mockResolvedValue("")
    mocks.exportRunsCsvMock.mockResolvedValue("")
    mocks.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({
      items: [{ id: 8, name: "Ransomware monitor" }],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [buildRun()], total: 1, has_more: false })
  })

  it("replaces the Activity table with run cards and preserves run actions at extension width", async () => {
    setViewport(420)
    render(<RunsTab />)

    expect(await screen.findByTestId("watchlists-runs-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table", { name: "Activity runs table" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show advanced filters" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Export CSV/ })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export options" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-runs-toolbar-actions")).toHaveClass("flex-wrap")
    expect(screen.getByTestId("runs-csv-export-button")).toHaveTextContent("Export CSV")
    expect(screen.getByTestId("runs-csv-export-button")).not.toHaveTextContent("Standard CSV")

    const card = screen.getByTestId("watchlists-run-card-55")
    expect(within(card).getByText("Ransomware monitor")).toBeInTheDocument()
    expect(within(card).getByText("Running")).toBeInTheDocument()
    expect(within(card).getByText("Found")).toBeInTheDocument()
    expect(within(card).getByText("12")).toBeInTheDocument()
    expect(within(card).getByText("Processed")).toBeInTheDocument()
    expect(within(card).getByText("8")).toBeInTheDocument()
    expect(within(card).getByText("Filtered")).toBeInTheDocument()
    expect(within(card).getByText("3")).toBeInTheDocument()
    expect(within(card).getByText("Errors")).toBeInTheDocument()
    expect(within(card).getByText("1")).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "View Details" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Open Reports" })).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Cancel run" })).toBeInTheDocument()

    fireEvent.click(within(card).getByRole("button", { name: "Open Reports" }))
    expect(mocks.storeStateRef.current.setOutputsJobFilter).toHaveBeenCalledWith(8)
    expect(mocks.storeStateRef.current.setOutputsRunFilter).toHaveBeenCalledWith(55)
    expect(mocks.storeStateRef.current.setActiveTab).toHaveBeenCalledWith("outputs")
  })

  it("preserves the desktop Activity table above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<RunsTab />)

    await waitFor(() => {
      expect(screen.getByRole("table", { name: "Activity runs table" })).toBeInTheDocument()
    })
    expect(screen.queryByTestId("watchlists-runs-constrained-list")).not.toBeInTheDocument()
    expect(screen.getByTestId("runs-csv-export-button")).toHaveTextContent("Export CSV (Standard CSV)")
  })
})

describe("RunDetailDrawer constrained management", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mocks.storeStateRef.current = baseState()
    mocks.getRunDetailsMock.mockResolvedValue({
      id: 55,
      job_id: 8,
      status: "completed",
      started_at: "2026-05-15T12:00:00Z",
      finished_at: "2026-05-15T12:05:00Z",
      stats: {
        items_found: 1,
        items_ingested: 1,
        items_filtered: 0,
        items_errored: 0
      },
      filter_tallies: {},
      error_msg: null,
      log_text: null,
      log_path: null,
      truncated: false,
      filtered_sample: null
    })
    mocks.fetchScrapedItemsMock.mockResolvedValue({
      items: [
        {
          id: 501,
          run_id: 55,
          job_id: 8,
          source_id: 13,
          url: "https://vendor.example/advisory",
          title: "Advisory update",
          summary: "Exploit activity observed.",
          content: "Body",
          published_at: "2026-05-15T11:50:00Z",
          tags: [],
          status: "ingested",
          reviewed: false,
          created_at: "2026-05-15T12:01:00Z"
        }
      ],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [{ id: 13, name: "Vendor feed" }],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistOutputsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.updateScrapedItemMock.mockResolvedValue({ reviewed: true })
  })

  it("renders run detail scraped items as constrained cards instead of a table", async () => {
    setViewport(420)
    render(<RunDetailDrawer open runId={55} onClose={vi.fn()} />)

    expect(await screen.findByTestId("watchlists-run-items-constrained-list")).toBeInTheDocument()
    expect(screen.queryByRole("table")).not.toBeInTheDocument()
    const card = screen.getByTestId("watchlists-run-item-card-501")
    expect(within(card).getByText("Advisory update")).toBeInTheDocument()
    expect(within(card).getByText("Exploit activity observed.")).toBeInTheDocument()
    expect(within(card).getByText("Vendor feed")).toBeInTheDocument()
    expect(within(card).getByText("Included in briefing")).toBeInTheDocument()
    expect(within(card).getByRole("button", { name: "Toggle reviewed for Advisory update" })).toBeInTheDocument()
    expect(within(card).getByRole("link", { name: "Open source" })).toHaveAttribute(
      "href",
      "https://vendor.example/advisory"
    )
  })
})
