import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"

const mocks = vi.hoisted(() => ({
  fetchJobRunsMock: vi.fn(),
  exportRunsCsvMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> }
}))

vi.mock("react-i18next", () => {
  const t = (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return _key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
  return { useTranslation: () => ({ t }) }
})

vi.mock("antd", () => {
  const Select = ({ value, onChange, options = [], allowClear, ...rest }: any) => (
    <select
      data-testid={rest["data-testid"] || "antd-select"}
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const next = event.currentTarget.value
        onChange?.(next === "" ? null : next)
      }}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)} disabled={Boolean(option.disabled)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Button = ({ children, onClick, loading, ...rest }: any) => (
    <button type="button" disabled={Boolean(loading)} onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  )

  const Table = ({ "aria-label": ariaLabel }: any) => (
    <div data-testid="runs-table" aria-label={ariaLabel} />
  )

  return {
    Select,
    Button,
    Dropdown: ({ children }: any) => <>{children}</>,
    Table,
    Progress: () => <div />,
    Tooltip: ({ children }: any) => <>{children}</>,
    Space: ({ children }: any) => <>{children}</>,
    Alert: ({ title, description, action }: any) => (
      <div>
        <div>{title}</div>
        <div>{description}</div>
        {action}
      </div>
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
  fetchJobRuns: (...args: any[]) => mocks.fetchJobRunsMock(...args),
  exportRunsCsv: (...args: any[]) => mocks.exportRunsCsvMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistRuns: (...args: any[]) => mocks.fetchWatchlistRunsMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: any) => unknown) => selector(mocks.storeStateRef.current)
}))

vi.mock("../RunDetailDrawer", () => ({
  RunDetailDrawer: () => null
}))

const buildRun = (status: string) => ({
  id: 7,
  job_id: 3,
  status,
  started_at: "2026-02-23T08:00:00Z",
  finished_at: status === "completed" ? "2026-02-23T08:05:00Z" : null,
  stats: {
    items_found: 10,
    items_ingested: status === "completed" ? 10 : 2
  }
})

const baseState = (overrides: Record<string, unknown> = {}) => ({
  runs: [],
  runsLoading: false,
  runsTotal: 0,
  runsPage: 1,
  runsPageSize: 20,
  runsJobFilter: null,
  runsStatusFilter: null,
  pollingActive: false,
  runDetailOpen: false,
  selectedRunId: null,
  setRuns: vi.fn(),
  setRunsLoading: vi.fn(),
  setRunsPage: vi.fn(),
  setRunsPageSize: vi.fn(),
  setRunsJobFilter: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setPollingActive: vi.fn(),
  openRunDetail: vi.fn(),
  closeRunDetail: vi.fn(),
  updateRunInList: vi.fn(),
  ...overrides
})

describe("RunsTab accessibility live-region behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.storeStateRef.current = baseState({ runs: [buildRun("running")], runsTotal: 1 })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchJobRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistJobsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.exportRunsCsvMock.mockResolvedValue("")
  })

  it("announces visible run status transitions in the SR live region", async () => {
    const { rerender } = render(<RunsTab />)
    expect(screen.getByTestId("watchlists-runs-live-region")).toHaveTextContent("")
    expect(screen.getByTestId("watchlists-runs-live-region")).toHaveAttribute("role", "status")
    expect(screen.getByTestId("watchlists-runs-live-region")).toHaveAttribute("aria-live", "polite")
    expect(screen.getByTestId("watchlists-runs-live-region")).toHaveAttribute("aria-atomic", "true")

    mocks.storeStateRef.current = baseState({
      runs: [buildRun("completed")],
      runsTotal: 1
    })
    rerender(<RunsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-runs-live-region")).toHaveTextContent(
        "Run #7 status changed to Completed."
      )
    })
  })

  it("provides an explicit table label for screen readers", () => {
    render(<RunsTab />)
    expect(screen.getByTestId("runs-table")).toHaveAttribute("aria-label", "Activity runs table")
  })

  it("rechecks stalled runs on the existing polling interval", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-02-23T08:44:59Z"))
    const runs = [buildRun("running")]
    mocks.storeStateRef.current = baseState({
      runs,
      runsTotal: 1,
      pollingActive: true,
      activeTab: "runs"
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: runs, total: 1 })
    const view = render(<RunsTab />)
    try {
      await act(async () => {})
      expect(screen.queryByText(/0 failed runs and 1 stalled run need review/)).not.toBeInTheDocument()
      await act(async () => vi.advanceTimersByTimeAsync(5000))
      expect(screen.getByText(/0 failed runs and 1 stalled run need review/)).toBeInTheDocument()
      view.unmount()
      const fetchCount = mocks.fetchWatchlistRunsMock.mock.calls.length
      await act(async () => vi.advanceTimersByTimeAsync(5000))
      expect(mocks.fetchWatchlistRunsMock).toHaveBeenCalledTimes(fetchCount)
    } finally {
      view.unmount()
      vi.useRealTimers()
    }
  })

  it("detects a run that becomes stalled while an unchanged response is pending", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-02-23T08:44:59Z"))
    const runs = [buildRun("running")]
    const response = { items: runs, total: 1 }
    let resolveRequest!: (value: typeof response) => void
    mocks.fetchWatchlistRunsMock.mockReturnValue(new Promise<typeof response>((resolve) => {
      resolveRequest = resolve
    }))
    mocks.storeStateRef.current = baseState({ runs, runsTotal: 1 })
    const view = render(<RunsTab />)
    try {
      await act(async () => {})
      expect(screen.queryByText(/0 failed runs and 1 stalled run need review/)).not.toBeInTheDocument()
      await act(async () => vi.advanceTimersByTimeAsync(2000))
      await act(async () => resolveRequest(response))
      expect(screen.getByText(/0 failed runs and 1 stalled run need review/)).toBeInTheDocument()
      expect(mocks.fetchWatchlistRunsMock).toHaveBeenCalledTimes(1)
    } finally {
      view.unmount()
      vi.useRealTimers()
    }
  })
})
